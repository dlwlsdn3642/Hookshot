"""Fork-wide SIMD fast paths for translated AArch64 NEON floating point.

The stock lifter already vectorizes FADD/FSUB and 4S FMUL, but emits one
scalar helper call per lane for FMLA, FDIV, and FSQRT. Replace only *exactly
recognized* A64 instruction translation shapes with SSE implementations.
FMLA matches this fork's existing native-FMA behavior; FDIV/FSQRT use the
upstream FPCR/FPSR fast gate and conservative positive-normal checks, and
fall back to the same lane helpers on every exceptional case.

Only 4S FMLA/FMLS and FSQRT are rewritten by default: 2S helpers
were slower and FDIV was neutral in a synthetic x86-64-v3 benchmark.
4S FDIV is opt-in via A64_NATIVE_VECTOR_DIV=1. All data is loaded before
writing the destination, including when
source and destination A64 registers alias. 2S operations zero the high D
half as the A64 translator does. Other instructions remain unchanged.
"""
from __future__ import annotations

import os
import re

import a64_lift

_BASE_OP_TEXT = None  # captured when install() executes, after native_fma.install()
_BASE_FPX_HEADER = None
_INSTALLED = False


def _compact(text):
    """Only for matching a single generated ARM instruction, not C rewriting."""
    return re.sub(r"\s+", "", text)


# The translator's exact per-lane FMLA shape (already outlined to the
# native scalar a64_fpx_fma32_fma32 by the underlying a64_lift.op_text).
_FMA = re.compile(
    r"\{uint32_t_n\[(?P<k>[24])\],_m\[(?P=k)\],_d\[(?P=k)\];"
    r"memcpy\(_n,c->vreg\[(?P<a>\d+)\],(?P<size>8|16)\);"
    r"memcpy\(_m,c->vreg\[(?P<b>\d+)\],(?P=size)\);"
    r"memcpy\(_d,c->vreg\[(?P<d>\d+)\],(?P=size)\);"
    r"for\(unsigned_j=0;_j<(?P=k);\+\+_j\)"
    r"\{uint64_t_a=_n\[_j\],_b=_m\[_j\],_z=_d\[_j\],_v;"
    r"(?P<neg>_a\^=0x80000000ULL;)?"
    r"_v=a64_fpx_fma32_fma32\(c,_a,_b,_z\);"
    r"_d\[_j\]=\(uint32_t\)_v;\}"
    r"(?P<store>.*)\}",
)
# Indexed FMLA/FMLS (vD.2S/.4S, vA, vB.s[lane]): the translator
# broadcasts one S lane through 2/4 scalar native FMA helper calls.
# FMLS negates A's sign bit before fused accumulation.
_FMA_LANE = re.compile(
    r"\{uint32_t_a\[(?P<k>[24])\],_r\[(?P=k)\],_m;"
    r"memcpy\(_a,c->vreg\[(?P<a>\d+)\],(?P<size>8|16)\);"
    r"memcpy\(&_m,\(constuint8_t\*\)c->vreg\[(?P<b>\d+)\]\+(?P<offset>\d+),4\);"
    r"memcpy\(_r,c->vreg\[(?P<d>\d+)\],(?P=size)\);"
    r"for\(int_i=0;_i<(?P=k);\+\+_i\)"
    r"\{uint64_t_n=_a\[_i\],_z0=_r\[_i\];"
    r"\{uint64_t_a=_n,_b=_m,_z=_z0,_v;"
    r"(?P<neg>_a\^=0x80000000ULL;)?"
    r"_v=a64_fpx_fma32_fma32\(c,_a,_b,_z\);"
    r"_r\[_i\]=\(uint32_t\)_v;\}\}"
    r"(?P<store>.*)\}",
)


def _check_lane_store(m):
    k, size, dest = int(m["k"]), int(m["size"]), m["d"]
    offset = int(m["offset"])
    if size != k * 4 or offset % 4 or offset > 12:
        return False
    if k == 4:
        return m["store"] == (
            f"a64_vset64(c->vreg[{dest}],0);a64_vst16(c->vreg[{dest}],_r);"
        )
    return m["store"] == f"a64_vset_lo(c->vreg[{dest}],_r,8);"


# FDIV's 2S and 4S translations have matching layout but no old destination.
_DIV = re.compile(
    r"\{uint32_t_n\[(?P<k>[24])\],_m\[(?P=k)\],_d\[(?P=k)\];"
    r"memcpy\(_n,c->vreg\[(?P<a>\d+)\],(?P<size>8|16)\);"
    r"memcpy\(_m,c->vreg\[(?P<b>\d+)\],(?P=size)\);"
    r"for\(unsigned_j=0;_j<(?P=k);\+\+_j\)"
    r"\{uint64_t_a=_n\[_j\],_b=_m\[_j\],_v;"
    r"_v=a64_fpx_div32_div32\(c,_a,_b\);"
    r"_d\[_j\]=\(uint32_t\)_v;\}"
    r"(?P<store>.*)\}",
)

# FSQRT's original translator always uses 4-element scratch arrays and
# initializes _r to zero, so even 2S writes zero into the high D half.
_SQRT = re.compile(
    r"\{uint32_t_n\[4\],_m\[4\],_r\[4\]=\{0\};"
    r"memcpy\(_n,c->vreg\[(?P<a>\d+)\],16\);"
    r"memcpy\(_m,c->vreg\[(?P<b>\d+)\],16\);"
    r"for\(unsigned_lane=0;_lane<(?P<k>[24]);\+\+_lane\)"
    r"\{uint64_t_a=_n\[_lane\],_v=0;"
    r"_v=a64_fpx_sqrt32_sqrt32\(c,_a\);"
    r"_r\[_lane\]=\(uint32_t\)\(_v\);\}"
    r"a64_vst16\(c->vreg\[(?P<d>\d+)\],_r\);\}",
)


def _dest(m):
    if "d" in m.re.groupindex:
        return m["d"]
    found = re.search(r"c->vreg\[(\d+)\]", m["store"])
    return found.group(1) if found else None


def _check_store(m):
    k, sz, d = int(m["k"]), int(m["size"]), _dest(m)
    if sz != k * 4:
        return False
    tail = m["store"]
    if k == 4:
        return tail == f"a64_vst16(c->vreg[{d}],_d);"
    return tail == f"memcpy(c->vreg[{d}],_d,8);c->vreg[{d}][1]=0;"


def replace_op(text):
    """Replace a single generated C statement, conservatively and atomically."""
    if os.environ.get("A64_NO_NATIVE_VECTOR"):
        return text
    compact = _compact(text)
    for kind, rx in (("fma", _FMA), ("div", _DIV)):
        # Vector FDIV was ~neutral in a synthetic CPU benchmark. Keep it
        # opt-in until full game frame-time comparisons establish a gain.
        if kind == "div" and os.environ.get("A64_NATIVE_VECTOR_DIV") != "1":
            continue
        m = rx.fullmatch(compact)
        if m and m["k"] == "4" and _check_store(m):
            actual = "fms" if kind == "fma" and m.groupdict().get("neg") else kind
            return (
                f"    a64_nv_{actual}32(c,c->vreg[{_dest(m)}],"
                f"c->vreg[{m['a']}],c->vreg[{m['b']}],{m['k']});"
            )
    lane = _FMA_LANE.fullmatch(compact)
    if lane and lane["k"] == "4" and _check_lane_store(lane):
        return (f"    a64_nv_fma32_lane(c,c->vreg[{lane['d']}],"
                f"c->vreg[{lane['a']}],c->vreg[{lane['b']}],"
                f"{int(lane['offset']) // 4},{lane['k']},"
                f"{int(bool(lane['neg']))});")
    m = _SQRT.fullmatch(compact)
    if m and m["k"] == "4":
        return f"    a64_nv_sqrt32(c,c->vreg[{m['d']}],c->vreg[{m['a']}],{m['k']});"
    return text


# The macro FPX wrappers remain available for uncommon FP cases. There are
# no new game-specific constants, addresses, or image-derived data here.
HELPER_HEADER = r"""
/* Fork NEON vector fast paths; generated by nx2/native_vector.py.
   Normal FP32 FDIV/FSQRT can use SSE when ARM FPCR/FPSR allow host FP.
   All rejected lanes execute the existing exact-capable scalar wrapper. */
#if defined(__SSE2__) && RECOMP_FPX_HOST == 2
#include <immintrin.h>
#endif

#if defined(A64_NV_HAS_FMA32)
static __attribute__((noinline)) void
a64_nv_fma32(Ctx *A64_NOESC c, uint64_t *A64_NOESC dst, const uint64_t *A64_NOESC pa, const uint64_t *A64_NOESC pb, unsigned lanes) {
    uint32_t a[4] = {0}, b[4] = {0}, z[4] = {0};
    memcpy(a, pa, lanes * 4u);
    memcpy(b, pb, lanes * 4u);
    memcpy(z, dst, lanes * 4u);
#if defined(__FMA__) && defined(__SSE2__) && RECOMP_FPX_HOST == 2
    /* Exceptional NaNs, infinities, and subnormals take the original
       scalar hardware FMA: preserve its operand/NaN behavior exactly. */
    /* SSE2 integer mask: normal and subnormal inputs need no per-lane
       scalar checks. Nonfinite inputs keep the original scalar helper's
       NaN payload/exceptional behavior (especially zero times infinity). */
    const __m128i am = _mm_set1_epi32(0x7fffffff);
    const __m128i im = _mm_set1_epi32(0x7f7fffff);
    const __m128i av = _mm_loadu_si128((const __m128i *)a);
    const __m128i bv = _mm_loadu_si128((const __m128i *)b);
    const __m128i zv = _mm_loadu_si128((const __m128i *)z);
    const __m128i invalid = _mm_or_si128(
        _mm_or_si128(_mm_cmpgt_epi32(_mm_and_si128(av, am), im),
                     _mm_cmpgt_epi32(_mm_and_si128(bv, am), im)),
        _mm_cmpgt_epi32(_mm_and_si128(zv, am), im));
    unsigned normal = !_mm_movemask_epi8(invalid);
    if (RECOMP_LIKELY(normal)) {
        const __m128 va = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)a));
        const __m128 vb = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)b));
        const __m128 vz = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)z));
        const __m128 vr = _mm_fmadd_ps(va, vb, vz);
        _mm_storeu_si128((__m128i *)z, _mm_castps_si128(vr));
    } else
#endif
    {
        for (unsigned j = 0; j < lanes; ++j)
            z[j] = (uint32_t)a64_fpx_fma32_fma32(c, a[j], b[j], z[j]);
    }
    if (lanes == 4) memcpy(dst, z, 16);
    else { memcpy(dst, z, 8); dst[1] = 0; }
}

/* Indexed NEON FMLA / FMLS: broadcast the selected source S lane, with
   subtraction implemented by flipping A's sign exactly as the translator.
   A, B and D are snapshotted before changing the destination. */
static __attribute__((noinline)) void
a64_nv_fma32_lane(Ctx *A64_NOESC c, uint64_t *A64_NOESC dst,
                  const uint64_t *A64_NOESC pa, const uint64_t *A64_NOESC pb,
                  unsigned lane, unsigned lanes, unsigned subtract) {
    uint32_t a[4] = {0}, z[4] = {0}, scalar;
    memcpy(a, pa, lanes * 4u);
    memcpy(z, dst, lanes * 4u);
    memcpy(&scalar, (const uint8_t *)pb + lane * 4u, 4);
    if (subtract)
        for (unsigned j = 0; j < lanes; ++j) a[j] ^= 0x80000000u;
#if defined(__FMA__) && defined(__SSE2__) && RECOMP_FPX_HOST == 2
    const __m128i am = _mm_set1_epi32(0x7fffffff);
    const __m128i im = _mm_set1_epi32(0x7f7fffff);
    const __m128i av = _mm_loadu_si128((const __m128i *)a);
    const __m128i zv = _mm_loadu_si128((const __m128i *)z);
    const __m128i invalid = _mm_or_si128(
        _mm_cmpgt_epi32(_mm_and_si128(av, am), im),
        _mm_cmpgt_epi32(_mm_and_si128(zv, am), im));
    unsigned normal = ((scalar & 0x7fffffffu) < 0x7f800000u) &&
                      !_mm_movemask_epi8(invalid);
    if (RECOMP_LIKELY(normal)) {
        const __m128 va = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)a));
        const __m128 vb = _mm_castsi128_ps(_mm_set1_epi32((int)scalar));
        const __m128 vz = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)z));
        const __m128 vr = _mm_fmadd_ps(va, vb, vz);
        _mm_storeu_si128((__m128i *)z, _mm_castps_si128(vr));
    } else
#endif
    {
        for (unsigned j = 0; j < lanes; ++j)
            z[j] = (uint32_t)a64_fpx_fma32_fma32(c, a[j], scalar, z[j]);
    }
    if (lanes == 4) memcpy(dst, z, 16);
    else { memcpy(dst, z, 8); dst[1] = 0; }
}

/* Full-vector FMLS: sign-flip A and reuse fused SIMD acceleration.
   Snapshot A before writing D to handle overlapping guest registers. */
static __attribute__((noinline)) void
a64_nv_fms32(Ctx *A64_NOESC c, uint64_t *A64_NOESC dst,
             const uint64_t *A64_NOESC pa, const uint64_t *A64_NOESC pb,
             unsigned lanes) {
    uint32_t negated[4] = {0};
    memcpy(negated, pa, lanes * 4u);
    for (unsigned j = 0; j < lanes; ++j)
        negated[j] ^= 0x80000000u;
    a64_nv_fma32(c, dst, (const uint64_t *)negated, pb, lanes);
}

#endif /* A64_NV_HAS_FMA32 */

/* Requiring finite normal inputs and outputs is more conservative than
   RECOMP_FPX_KEEPDIV32 but guarantees the same result and no changed FPSR.
   Mask the unused 2S lanes to 1.0f so the SSE instruction cannot generate
   spurious 0/0 invalid operations in ignored elements. */
#if defined(A64_NV_HAS_DIV32)
static __attribute__((noinline)) void
a64_nv_div32(Ctx *A64_NOESC c, uint64_t *A64_NOESC dst, const uint64_t *A64_NOESC pa, const uint64_t *A64_NOESC pb, unsigned lanes) {
    uint32_t a[4] = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u};
    uint32_t b[4] = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u};
    uint32_t z[4] = {0};
    memcpy(a, pa, lanes * 4u);
    memcpy(b, pb, lanes * 4u);
#if defined(__SSE2__) && RECOMP_FPX_HOST == 2
    if (RECOMP_LIKELY(RECOMP_FPX_OPEN(c))) {
        unsigned normal = 1;
        for (unsigned j = 0; j < lanes; ++j) {
            const uint32_t ea = a[j] & 0x7f800000u;
            const uint32_t eb = b[j] & 0x7f800000u;
            normal &= ea > 0 && ea < 0x7f800000u &&
                      eb > 0 && eb < 0x7f800000u;
        }
        if (RECOMP_LIKELY(normal)) {
            const __m128 va = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)a));
            const __m128 vb = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)b));
            const __m128 vr = _mm_div_ps(va, vb);
            _mm_storeu_si128((__m128i *)z, _mm_castps_si128(vr));
            unsigned good = 1;
            for (unsigned j = 0; j < lanes; ++j) {
                const uint32_t ez = z[j] & 0x7f800000u;
                good &= ez > 0 && ez < 0x7f800000u;
            }
            if (RECOMP_LIKELY(good)) {
            if (lanes == 4) memcpy(dst, z, 16);
            else { memcpy(dst, z, 8); dst[1] = 0; }
                return;
            }
        }
    }
#endif
    for (unsigned j = 0; j < lanes; ++j)
        z[j] = (uint32_t)a64_fpx_div32_div32(c, a[j], b[j]);
    if (lanes == 4) memcpy(dst, z, 16);
    else { memcpy(dst, z, 8); dst[1] = 0; }
}

#endif /* A64_NV_HAS_DIV32 */

#if defined(A64_NV_HAS_SQRT32)
static __attribute__((noinline)) void
a64_nv_sqrt32(Ctx *A64_NOESC c, uint64_t *A64_NOESC dst, const uint64_t *A64_NOESC pa, unsigned lanes) {
    uint32_t a[4] = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u};
    uint32_t z[4] = {0};
    memcpy(a, pa, lanes * 4u);
#if defined(__SSE2__) && RECOMP_FPX_HOST == 2
    if (RECOMP_LIKELY(RECOMP_FPX_OPEN(c))) {
        unsigned good = 1;
        for (unsigned j = 0; j < lanes; ++j) {
            const uint32_t ea = a[j] & 0x7f800000u;
            good &= !(a[j] & 0x80000000u) &&
                    ea > 0 && ea < 0x7f800000u;
        }
        if (RECOMP_LIKELY(good)) {
            const __m128 va = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)a));
            const __m128 vr = _mm_sqrt_ps(va);
            _mm_storeu_si128((__m128i *)z, _mm_castps_si128(vr));
            if (lanes == 4) memcpy(dst, z, 16);
            else { memcpy(dst, z, 8); dst[1] = 0; }
            return;
        }
    }
#endif
    for (unsigned j = 0; j < lanes; ++j)
        z[j] = (uint32_t)a64_fpx_sqrt32_sqrt32(c, a[j]);
    memcpy(dst, z, 16);
}
#endif /* A64_NV_HAS_SQRT32 */
"""


def install():
    """Hook the generator after native_fma.install(), preserving its wrappers."""
    global _INSTALLED, _BASE_OP_TEXT, _BASE_FPX_HEADER
    if _INSTALLED:
        return
    # Imports happen before the original native_fma.install() call in lift.py.
    # Capturing fpx_header during module import accidentally *undoes* that
    # native-FMA override for all non-vector operations.
    _BASE_OP_TEXT = a64_lift.op_text
    _BASE_FPX_HEADER = a64_lift.fpx_header
    _INSTALLED = True

    def op_text(source, used):
        return replace_op(_BASE_OP_TEXT(source, used))

    def fpx_header(used):
        result = _BASE_FPX_HEADER(used)
        if os.environ.get("A64_NO_NATIVE_VECTOR"):
            return result
        variants = {
            ("fma32", "fma32", 3, 3): "A64_NV_HAS_FMA32",
            ("div32", "div32", 2, 2): "A64_NV_HAS_DIV32",
            ("sqrt32", "sqrt32", 1, 1): "A64_NV_HAS_SQRT32",
        }
        active = [name for shape, name in variants.items() if shape in used]
        if not active:
            return result
        return result + "".join("#define %s 1\n" % name for name in active) + HELPER_HEADER

    a64_lift.op_text = op_text
    a64_lift.fpx_header = fpx_header
