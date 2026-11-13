"""Cross-check fork-wide SIMD paths against the old scalar-lane dispatch.

This test uses only synthetic ARM-to-C instruction shapes and a standalone C
harness. No game dump, game-generated C, or game assets are needed.
"""
import ctypes
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import kit  # noqa: F401,E402 (registers a64-kit on sys.path)
import a64_lift  # noqa: E402
import native_fma  # noqa: E402
import native_vector as nv  # noqa: E402


def generated_shape(op, lanes):
    """One generic, non-game-specific form of each translator instruction."""
    size = lanes * 4
    if op in ("fma", "fms", "div"):
        inputs = (f"{{uint32_t _n[{lanes}],_m[{lanes}],_d[{lanes}];"
                  f"memcpy(_n,c->vreg[1],{size});"
                  f"memcpy(_m,c->vreg[2],{size});")
        if op in ("fma", "fms"):
            inputs += f"memcpy(_d,c->vreg[0],{size});"
            lane = ("uint64_t _a=_n[_j],_b=_m[_j],_z=_d[_j],_v;"
                    + ("_a^=0x80000000ULL;" if op == "fms" else "")
                    + "_v=a64_fpx_fma32_fma32(c,_a,_b,_z);")
        else:
            lane = ("uint64_t _a=_n[_j],_b=_m[_j],_v;"
                    "_v=a64_fpx_div32_div32(c,_a,_b);")
        inputs += (f"for(unsigned _j=0;_j<{lanes};++_j)"
                   "{" + lane + "_d[_j]=(uint32_t)_v;}")
        if lanes == 4:
            return inputs + "a64_vst16(c->vreg[0],_d);}"
        return inputs + "memcpy(c->vreg[0],_d,8);c->vreg[0][1]=0;}"
    return ("{uint32_t _n[4],_m[4],_r[4]={0};"
            "memcpy(_n,c->vreg[1],16);memcpy(_m,c->vreg[2],16);"
            f"for(unsigned _lane=0;_lane<{lanes};++_lane)"
            "{uint64_t _a=_n[_lane],_v=0;"
            "_v=a64_fpx_sqrt32_sqrt32(c,_a);_r[_lane]=(uint32_t)(_v);}"
            "a64_vst16(c->vreg[0],_r);}")



def generated_lane_shape(lanes, lane, subtract):
    """The synthetic translator shape for indexed vector FMLA/FMLS."""
    size = lanes * 4
    prefix = (
        f"{{uint32_t _a[{lanes}],_r[{lanes}],_m;"
        f"memcpy(_a,c->vreg[1],{size});"
        f"memcpy(&_m,(const uint8_t*)c->vreg[2]+{lane*4},4);"
        f"memcpy(_r,c->vreg[0],{size});"
        f"for(int _i=0;_i<{lanes};++_i)"
        "{uint64_t _n=_a[_i],_z0=_r[_i];"
        "{uint64_t _a=_n,_b=_m,_z=_z0,_v;"
    )
    prefix += ("_a^=0x80000000ULL;" if subtract else "")
    prefix += ("_v=a64_fpx_fma32_fma32(c,_a,_b,_z);"
               "_r[_i]=(uint32_t)_v;}}")
    if lanes == 4:
        return prefix + "a64_vset64(c->vreg[0],0);a64_vst16(c->vreg[0],_r);}"
    return prefix + "a64_vset_lo(c->vreg[0],_r,8);}"



def check_generator():
    native_fma.install()
    nv.install()
    nv.install()
    assert nv._BASE_FPX_HEADER is native_fma.fpx_header, (
        "native_vector must retain the preinstalled native-FMA generator"
    )
    assert '__builtin_fmaf' in a64_lift.fpx_header({("fma32", "fma32", 3, 3)})
    for op in ("fma", "fms", "div", "sqrt"):
        for lanes in (2, 4):
            original = generated_shape(op, lanes)
            converted = a64_lift.op_text(original, set())
            assert converted == (
                f"    a64_nv_{op}32(c,c->vreg[0],c->vreg[1],"
                + ("c->vreg[2]," if op != "sqrt" else "")
                + f"{lanes});"
            ), (op, lanes, converted)
            assert nv.replace_op(original + "return 0;") == original + "return 0;"
    for lanes in (2, 4):
        for lane in range(4):
            for subtract in (0, 1):
                original = generated_lane_shape(lanes, lane, subtract)
                final = a64_lift.op_text(original, set())
                assert final == (
                    f"    a64_nv_fma32_lane(c,c->vreg[0],c->vreg[1],"
                    f"c->vreg[2],{lane},{lanes},{subtract});"
                ), (lanes, lane, subtract, final)

    os.environ["A64_NO_NATIVE_VECTOR"] = "1"
    try:
        for op in ("fma", "fms", "div", "sqrt"):
            assert nv.replace_op(generated_shape(op, 4)) == generated_shape(op, 4)
        assert a64_lift.fpx_header({("fma32", "fma32", 3, 3)}) == nv._BASE_FPX_HEADER(
            {("fma32", "fma32", 3, 3)}
        )
    finally:
        del os.environ["A64_NO_NATIVE_VECTOR"]
    names = {
        "fma": ("fma32", "fma32", 3, 3),
        "fms": ("fma32", "fma32", 3, 3),
        "div": ("div32", "div32", 2, 2),
        "sqrt": ("sqrt32", "sqrt32", 1, 1),
    }
    # Only helpers backed by scalar wrappers in this build may be emitted.
    for op, shape in names.items():
        h = a64_lift.fpx_header({shape})
        assert h.count("#define A64_NV_HAS_") == 1, (op, h[-500:])


C_HEAD = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
typedef struct { uint64_t fpcr, fpsr; } Ctx;
#define A64_NOESC
#define RECOMP_FPX_HOST 2
#define RECOMP_LIKELY(x) (__builtin_expect(!!(x),1))
#define RECOMP_FPX_OPEN(c) (((c)->fpsr & 16u) && !((c)->fpcr & 0x06C8FF07u))
static unsigned fma_calls, div_calls, sqrt_calls;
void reset_calls(void) { fma_calls = div_calls = sqrt_calls = 0; }
unsigned count_calls(int op) {
    return op == 0 ? fma_calls : op == 1 ? div_calls : sqrt_calls;
}
static float frombits(uint32_t x) { float f; memcpy(&f,&x,4); return f; }
static uint32_t tobits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static __attribute__((noinline)) uint64_t a64_fpx_fma32_fma32(Ctx *c, uint64_t a, uint64_t b, uint64_t z) {
    (void)c; ++fma_calls;
    return tobits(__builtin_fmaf(frombits(a),frombits(b),frombits(z)));
}
static __attribute__((noinline)) uint64_t a64_fpx_div32_div32(Ctx *c, uint64_t a, uint64_t b) {
    ++div_calls;
    if (((uint32_t)b & 0x7fffffffu) == 0) c->fpsr |= 1;
    volatile float fa=frombits(a), fb=frombits(b);
    return tobits(fa/fb);
}
static __attribute__((noinline)) uint64_t a64_fpx_sqrt32_sqrt32(Ctx *c, uint64_t a) {
    ++sqrt_calls;
    if (((uint32_t)a & 0x80000000u) &&
        ((uint32_t)a & 0x7fffffffu)) c->fpsr |= 1;
    const __m128 v = _mm_sqrt_ss(_mm_set_ss(frombits(a)));
    return tobits(_mm_cvtss_f32(v));
}
#define A64_NV_HAS_FMA32 1
#define A64_NV_HAS_DIV32 1
#define A64_NV_HAS_SQRT32 1
"""

C_TAIL = r"""
void run(int op, int fast, int lanes, Ctx *ctx, uint64_t *dst,
         const uint64_t *pa, const uint64_t *pb) {
    if (fast) {
        if (op == 0) a64_nv_fma32(ctx,dst,pa,pb,(unsigned)lanes);
        else if (op == 1) a64_nv_div32(ctx,dst,pa,pb,(unsigned)lanes);
        else if (op == 3) a64_nv_fms32(ctx,dst,pa,pb,(unsigned)lanes);
        else a64_nv_sqrt32(ctx,dst,pa,(unsigned)lanes);
    } else {
        uint32_t a[4]={0}, b[4]={0}, z[4]={0};
        memcpy(a,pa,lanes*4);
        memcpy(b,pb,lanes*4);
        if (op==0 || op==3) memcpy(z,dst,lanes*4);
        for(int j=0;j<lanes;j++) {
            if(op==0 || op==3) {
                if(op==3) a[j] ^= 0x80000000u;
                z[j]=(uint32_t)a64_fpx_fma32_fma32(ctx,a[j],b[j],z[j]);
            }
            else if(op==1) z[j]=(uint32_t)a64_fpx_div32_div32(ctx,a[j],b[j]);
            else z[j]=(uint32_t)a64_fpx_sqrt32_sqrt32(ctx,a[j]);
        }
        memcpy(dst,z,16);
    }
}

void run_lane(int fast, int lanes, int lane, int subtract,
              Ctx *ctx, uint64_t *dst, const uint64_t *pa, const uint64_t *pb) {
    if (fast) {
        a64_nv_fma32_lane(ctx,dst,pa,pb,(unsigned)lane,
                          (unsigned)lanes,(unsigned)subtract);
    } else {
        uint32_t a[4]={0}, z[4]={0}, scalar;
        memcpy(a,pa,lanes*4);
        memcpy(z,dst,lanes*4);
        memcpy(&scalar,(const uint8_t*)pb + lane*4,4);
        for (int j=0; j<lanes; ++j) {
            if (subtract) a[j] ^= 0x80000000u;
            z[j] = (uint32_t)a64_fpx_fma32_fma32(ctx,a[j],scalar,z[j]);
        }
        memcpy(dst,z,16);
    }
}
"""


def compiled():
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("Native vector test requires a C compiler")
    with tempfile.TemporaryDirectory(prefix="hookshot-vector-") as td:
        src = Path(td) / "vector.c"
        out = Path(td) / ("vector.dll" if os.name == "nt" else "vector.so")
        src.write_text(C_HEAD + nv.HELPER_HEADER + C_TAIL, encoding="utf-8")
        cmd = [cc, "-shared", "-fPIC", "-O2", "-mfma", "-msse2",
               "-ffp-contract=off", "-fno-fast-math",
               "-o", str(out), str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError("C SIMD helper compile failed:\n" + proc.stderr[:3800])
        lib = ctypes.CDLL(str(out))
        lib.run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_void_p]
        lib.run_lane.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p]
        lib.count_calls.argtypes = [ctypes.c_int]
        lib.count_calls.restype = ctypes.c_uint
        regtype = ctypes.c_uint64 * 2

        def ptr(data):
            return ctypes.cast(data, ctypes.c_void_p)

        # Normal finite inputs exercise SIMD, specials and modified FPCR
        # force the old scalar helper (with its own FPSR side effects).
        patterns = [
            (0x3fa00000, 0x40000000, 0x40400000, 0x3f000000),
            (0x3f800000, 0x40000000, 0x3f400000, 0x41100000),
            (0x00000000, 0x80000000, 0x3f800000, 0x40000000),
            (0x7f800000, 0xff800000, 0x7fc00001, 0x7fa00001),
            (0x00000001, 0x80000001, 0x00800000, 0x7f7fffff),
        ]
        rng = random.Random(0x4E454F4E)
        trials = 0
        speedy = 0
        for op in range(4):
            for lanes in (2, 4):
                for case in range(450):
                    if case < len(patterns):
                        vals = patterns[case]
                    elif case % 4:
                        vals = tuple(0x3f000000 | (rng.getrandbits(24) & 0x00ffffff)
                                     for _ in range(4))
                    else:
                        vals = tuple(rng.getrandbits(32) for _ in range(4))
                    a = regtype()
                    b = regtype()
                    z = regtype()
                    ctypes.memmove(ptr(a), (ctypes.c_uint32*4)(*vals), 16)
                    bbits = tuple(rng.choice(vals) for _ in range(4))
                    ctypes.memmove(ptr(b), (ctypes.c_uint32*4)(*bbits), 16)
                    zbits = tuple(rng.choice(vals) for _ in range(4))
                    ctypes.memmove(ptr(z), (ctypes.c_uint32*4)(*zbits), 16)
                    for fpcr,fpsr in ((0,16),(0x01000000,0x9f),
                                      (0x06C8FF07,0),(0,0)):
                        for alias in (0, 1, 2):
                            ra, rb, rz = regtype(*a), regtype(*b), regtype(*z)
                            sa, sb, sz = regtype(*a), regtype(*b), regtype(*z)
                            if alias == 1:
                                rz = ra
                                sz = sa
                            elif alias == 2:
                                rz = rb
                                sz = sb
                            cx1=(ctypes.c_uint64*2)(fpcr,fpsr)
                            cx2=(ctypes.c_uint64*2)(fpcr,fpsr)
                            lib.reset_calls()
                            lib.run(op,1,lanes,ptr(cx1),ptr(rz),ptr(ra),ptr(rb))
                            fastcalls=lib.count_calls(op)
                            lib.reset_calls()
                            lib.run(op,0,lanes,ptr(cx2),ptr(sz),ptr(sa),ptr(sb))
                            assert bytes(rz)==bytes(sz), (op,lanes,case,fpcr,fpsr,alias,
                                bytes(rz).hex(),bytes(sz).hex())
                            assert cx1[1] == cx2[1], (op,lanes,case,fpcr,fpsr,alias)
                            assert fastcalls in (0,lanes), (op,lanes,fastcalls)
                            speedy += fastcalls == 0
                            trials+=1
        # Indexed FMLA/FMLS: all four broadcast lanes, both vector sizes,
        # both add/subtract encodings, FPSR/FPCR gates and source/dest alias.
        lane_trials = lane_speedy = 0
        for lanes in (2, 4):
            for selected in range(4):
                for subtract in (0, 1):
                    for case in range(170):
                        if case < len(patterns):
                            vals = patterns[case]
                        elif case % 4:
                            vals = tuple(0x3f000000 | (rng.getrandbits(24) & 0x00ffffff)
                                         for _ in range(4))
                        else:
                            vals = tuple(rng.getrandbits(32) for _ in range(4))
                        a, b, z = regtype(), regtype(), regtype()
                        ctypes.memmove(ptr(a), (ctypes.c_uint32*4)(*vals), 16)
                        bbits = tuple(rng.choice(vals) for _ in range(4))
                        ctypes.memmove(ptr(b), (ctypes.c_uint32*4)(*bbits), 16)
                        zbits = tuple(rng.choice(vals) for _ in range(4))
                        ctypes.memmove(ptr(z), (ctypes.c_uint32*4)(*zbits), 16)
                        for fpcr, fpsr in ((0, 16), (0x01000000, 0x9f),
                                           (0x06C8FF07, 0)):
                            for alias in (0, 1, 2):
                                ra, rb, rz = regtype(*a), regtype(*b), regtype(*z)
                                sa, sb, sz = regtype(*a), regtype(*b), regtype(*z)
                                if alias == 1:
                                    rz, sz = ra, sa
                                elif alias == 2:
                                    rz, sz = rb, sb
                                cx1 = (ctypes.c_uint64 * 2)(fpcr, fpsr)
                                cx2 = (ctypes.c_uint64 * 2)(fpcr, fpsr)
                                lib.reset_calls()
                                lib.run_lane(1, lanes, selected, subtract,
                                             ptr(cx1), ptr(rz), ptr(ra), ptr(rb))
                                f_calls = lib.count_calls(0)
                                lib.reset_calls()
                                lib.run_lane(0, lanes, selected, subtract,
                                             ptr(cx2), ptr(sz), ptr(sa), ptr(sb))
                                assert bytes(rz) == bytes(sz), (
                                    "indexed", lanes, selected, subtract, case,
                                    fpcr, fpsr, alias, bytes(rz).hex(),
                                    bytes(sz).hex()
                                )
                                assert cx1[1] == cx2[1], (
                                    lanes, selected, subtract, case, fpcr, fpsr, alias
                                )
                                assert f_calls in (0, lanes), f_calls
                                lane_speedy += f_calls == 0
                                lane_trials += 1
        print("PASS: %d indexed FMLA/FMLS tests (%d no scalar calls)"
              % (lane_trials, lane_speedy))

        print("PASS: %d SIMD-vs-scalar tests (%d cases without scalar calls)"
              % (trials,speedy))
        if os.name == "nt":
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            assert ctypes.windll.kernel32.FreeLibrary(lib._handle)


if __name__ == "__main__":
    check_generator()
    compiled()
