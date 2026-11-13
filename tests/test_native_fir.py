"""Game-free test for the OoT 1025-tap FIR native source hook.

Builds the actual generated C hook in a tiny guest-memory harness; checks
random and PCM16-like data at ring boundaries against an independent scalar
FMA reference, and tests the lifter's build/word guards.
"""
import ctypes
import importlib.util
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
            self.funcs = {native_fir.FUNCTION: (
                {pc: (("op",), w) for pc, w in native_fir.WORDS.items()}, set()
            )}
            self.after = {}
            self.no_local = set()

    class Image:
        def __init__(self, bad=False):
            self.bad = bad

        def word(self, pc):
            if self.bad and pc == native_fir.HOOK_AFTER:
                return 0
            return native_fir.WORDS[pc]

    import oot_patches
    lift = Mock()
    assert not native_fir.install(lift, Image(), build_id="another-version")
    assert not lift.after
    assert native_fir.install(lift, Image(), oot_patches.LIFE_BUILD_ID)
    assert lift.after[native_fir.HOOK_AFTER] == native_fir.HOOK
    assert native_fir.FUNCTION in lift.no_local
    assert native_fir.POST_LOOP in lift.funcs[native_fir.FUNCTION][1]
    try:
        native_fir.install(lift, Image(), oot_patches.LIFE_BUILD_ID)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Hook installed twice")
    try:
        native_fir.install(Mock(), Image(bad=True), oot_patches.LIFE_BUILD_ID)
    except RuntimeError:
        pass
    else:
        raise AssertionError("An unknown instruction was accepted")


def compiled():
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        print("SKIP native compiler not available")
        return
    head = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct { uint64_t x[32]; uint64_t vreg[32][2]; } Ctx;
static unsigned char *mem;
static int polls;
void set_mem(void *ptr) { mem = (unsigned char *)ptr; }
int get_polls(void) { return polls; }
void reset_polls(void) { polls = 0; }
#define A64_PTR(a) (mem + (uint64_t)(a))
#define A64_POLL(c) (++polls)
static void a64_vset64(uint64_t *v, uint64_t bits) { v[0]=bits; v[1]=0; }
uint32_t fast(Ctx *c) {
"""
    tail = r"""
L_8338f058:;
return (uint32_t)c->vreg[0][0];
}
uint32_t ref(const float *p, int idx) {
    float sum=0.0f;
    for (int k=0; k<1025; ++k) {
        sum=__builtin_fmaf(p[1025+idx],p[k],sum);
        idx = idx ? idx-1 : 1024;
    }
    uint32_t bits;
    memcpy(&bits,&sum,4);
    return bits;
}
"""
    with tempfile.TemporaryDirectory(prefix="hookshot-fir-") as td:
        c=Path(td)/"test.c"
        lib=Path(td)/("test.dll" if os.name=="nt" else "test.so")
        c.write_text(head+native_fir.HOOK+tail,encoding="utf-8")
        args=[cc,"-O2","-mfma","-shared","-fPIC","-ffp-contract=off","-o",str(lib),str(c)]
        proc=subprocess.run(args,capture_output=True,text=True)
        if proc.returncode:
            raise RuntimeError("FIR hook did not compile: "+proc.stderr[:2500])
        loaded=ctypes.CDLL(str(lib))
        loaded.set_mem.argtypes=[ctypes.c_void_p]
        loaded.fast.argtypes=[ctypes.c_void_p]
        loaded.fast.restype=ctypes.c_uint32
        loaded.ref.argtypes=[ctypes.c_void_p,ctypes.c_int]
        loaded.ref.restype=ctypes.c_uint32
        loaded.get_polls.restype=ctypes.c_int
        data=(ctypes.c_float*2050)()
        loaded.set_mem(ctypes.cast(data,ctypes.c_void_p))
        rng=random.Random(0x8338efe0)
        count=0
        for case in range(25):
            for i in range(2050):
                data[i]=(rng.uniform(-1.0,1.0) if case%3
                         else rng.randrange(-32768,32768)/32768.0)
            for idx in (0,1,2,16,127,256,512,1023,1024):
                cx=(ctypes.c_uint64*(32+64))()
                cx[0]=0
                cx[9]=0x1004
                cx[10]=idx
                cx[12]=idx
                loaded.reset_polls()
                got=loaded.fast(ctypes.cast(cx,ctypes.c_void_p))
                want=loaded.ref(ctypes.cast(data,ctypes.c_void_p),idx)
                assert got==want,(case,idx,got,want)
                assert cx[8]==1025 and cx[12]==idx and cx[13]==0x1004+4*((idx+1)%1025)
                assert loaded.get_polls()==16,loaded.get_polls()
                count+=1
        print("PASS",count,"native FIR: bitwise FMA result, wrap, guest registers and polls")
        cx=(ctypes.c_uint64*(32+64))()
        cx[12]=1025
        cx[8]=42
        assert loaded.fast(ctypes.cast(cx,ctypes.c_void_p))==0 and cx[8]==42
        print("PASS unexpected ring index uses unmodified slow fallback")
        if os.name == "nt":
            # Windows holds a loaded DLL open until explicitly freed.
            ctypes.windll.kernel32.FreeLibrary.argtypes=[ctypes.c_void_p]
            assert ctypes.windll.kernel32.FreeLibrary(loaded._handle) != 0


if __name__=="__main__":
    guards()
    compiled()
