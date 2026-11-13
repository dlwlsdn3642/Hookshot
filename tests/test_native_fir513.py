"""Verify both 513-tap microphone FIR loops without game image or generated C.

The hooks must preserve the sequential ARM FMADD result, circular buffer
index, final LDR S1/S2 register values and guest arithmetic flags.
"""
import ctypes
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import native_fir  # noqa: E402


def guards():
    class Mock:
        def __init__(self):
            b = {pc: (("op",), word) for pc, word in native_fir.FIR513_WORDS.items()}
            for _, resume in native_fir.FIR513_SITES:
                b[resume] = (("op",), native_fir.FIR513_WORDS[resume])
            self.funcs = {native_fir.FIR513_FUNCTION: (b, set())}
            self.after = {}
            self.no_local = set()

    class Image:
        def __init__(self, bad=0):
            self.bad = bad

        def word(self, pc):
            if pc == self.bad:
                return 0
            return native_fir.FIR513_WORDS[pc]

    import oot_patches
    l = Mock()
    assert not native_fir.install(l, Image(), "unrelated game build")
    assert not l.after
    assert native_fir.install(l, Image(), oot_patches.LIFE_BUILD_ID)
    assert native_fir.FIR513_FUNCTION in l.no_local
    for after, resume in native_fir.FIR513_SITES:
        assert resume in l.funcs[native_fir.FIR513_FUNCTION][1]
        assert l.after[after] == native_fir.fir513_hook(resume)
    try:
        native_fir.install(l, Image(), oot_patches.LIFE_BUILD_ID)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Double application accepted")

    for site, _ in native_fir.FIR513_SITES:
        v = Mock()
        try:
            native_fir.install(v, Image(bad=site), oot_patches.LIFE_BUILD_ID)
        except RuntimeError:
            assert not v.after and not v.no_local
        else:
            raise AssertionError("Unrecognized guest instruction accepted")


def compiled():
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        print("SKIP: C compiler unavailable")
        return

    head = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct {
    uint64_t x[32];
    uint64_t vreg[32][2];
    uint8_t n, z, c, v;
} Ctx;
static unsigned char *mem;
static int polls;
void set_mem(void *p) { mem = (unsigned char *)p; }
void reset_polls(void) { polls = 0; }
int get_polls(void) { return polls; }
#define A64_PTR(a) (mem + (uint64_t)(a))
#define A64_POLL(c) (++polls)
static void a64_vset64(uint64_t *v, uint64_t bits) { v[0] = bits; v[1] = 0; }
uint32_t fast_a(Ctx *c) {
"""
    mid = r"""
L_8338d934:;
return (uint32_t)c->vreg[0][0];
}
uint32_t fast_b(Ctx *c) {
"""
    tail = r"""
L_8338de88:;
return (uint32_t)c->vreg[0][0];
}
uint32_t ref(const float *data, int index) {
    float sum = 0.0f;
    for (int k=0; k<513; ++k) {
        sum = __builtin_fmaf(data[513+index], data[k], sum);
        index = index ? index - 1 : 512;
    }
    uint32_t bits;
    memcpy(&bits,&sum,4);
    return bits;
}
"""
    program = (head + native_fir.fir513_hook(0x8338D934) + mid
               + native_fir.fir513_hook(0x8338DE88) + tail)
    with tempfile.TemporaryDirectory(prefix="hookshot-fir513-") as td:
        c = Path(td) / "test.c"
        lib = Path(td) / ("test.dll" if os.name == "nt" else "test.so")
        c.write_text(program, encoding="utf-8")
        args = [cc, "-O2", "-mfma", "-ffp-contract=off", "-shared",
                "-fPIC", "-o", str(lib), str(c)]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError("FIR513 C hook compile failed: " + proc.stderr[:2500])
        dll = ctypes.CDLL(str(lib))
        dll.set_mem.argtypes = [ctypes.c_void_p]
        dll.ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        dll.ref.restype = ctypes.c_uint32
        for name in ("fast_a", "fast_b"):
            fn = getattr(dll, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_uint32
        dll.get_polls.restype = ctypes.c_int
        samples = (ctypes.c_float * 1026)()
        dll.set_mem(ctypes.addressof(samples))
        rng = random.Random(0x8338D510)
        count = 0
        indices = (0, 1, 2, 16, 127, 256, 384, 511, 512)

        for case in range(25):
            for k in range(1026):
                samples[k] = (rng.uniform(-1, 1) if case % 3
                              else rng.randrange(-32768, 32768) / 32768.0)
            for idx in indices:
                want = dll.ref(samples, idx)
                last = 0 if idx == 512 else idx + 1
                expected_s1 = struct.unpack("<I", struct.pack("<f", samples[512]))[0]
                expected_s2 = struct.unpack("<I", struct.pack("<f", samples[513+last]))[0]
                for name in ("fast_a", "fast_b"):
                    ctx = ctypes.create_string_buffer(1024)
                    struct.pack_into("<Q", ctx, 8*8, 0)  # X8, coeff base
                    struct.pack_into("<Q", ctx, 9*8, 0)  # X9, zero offset
                    struct.pack_into("<Q", ctx, 10*8, 0x804)  # X10 ring base
                    struct.pack_into("<Q", ctx, 11*8, idx)
                    struct.pack_into("<Q", ctx, 12*8, idx)
                    dll.reset_polls()
                    got = getattr(dll, name)(ctypes.addressof(ctx))
                    x9, x12, x13 = struct.unpack_from("<QQQ", ctx, 9*8)[0], \
                        struct.unpack_from("<Q", ctx, 12*8)[0], \
                        struct.unpack_from("<Q", ctx, 13*8)[0]
                    v0, v1, v2 = struct.unpack_from("<Q", ctx, 256)[0], \
                        struct.unpack_from("<Q", ctx, 272)[0], \
                        struct.unpack_from("<Q", ctx, 288)[0]
                    assert (got, v0, v1, v2) == (want, want, expected_s1, expected_s2), \
                        (name, case, idx, got, want)
                    assert (x9, x12, x13) == (0x804, idx, 0x804 + last*4)
                    assert ctx.raw[768:772] == bytes((0, 1, 1, 0))
                    assert dll.get_polls() == 8
                    count += 1

        for name in ("fast_a", "fast_b"):
            for invalid in (513, 0xFFFFFFFF):
                ctx = ctypes.create_string_buffer(1024)
                struct.pack_into("<Q", ctx, 9*8, 42)
                struct.pack_into("<Q", ctx, 12*8, invalid)
                assert getattr(dll, name)(ctypes.addressof(ctx)) == 0
                assert struct.unpack_from("<Q", ctx, 9*8)[0] == 42

        print("PASS: %d 513-tap cases, bit-exact sum, ring state, SIMD lows, flags, polls"
              % count)
        if os.name == "nt":
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            assert ctypes.windll.kernel32.FreeLibrary(dll._handle)


if __name__ == "__main__":
    guards()
    compiled()
