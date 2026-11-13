"""Reduce the OoT microphone's 1025- and 513-tap FIR loops to native scalar FMA.

The guest function f_8338efe0 converts a PCM sample to float, inserts it in a
1025-float ring, convolves the ring with 1025 float coefficients, then clips and
converts the result back to PCM16. The lifted loop spills A64 registers,
branches, and calls a helper for every tap; this hook replaces ONLY the dot
product. The input insertion, ring-index update, clipping and return stay in
the lifted code. This is intentionally limited to the verified OoT build.

The scalar fused accumulation preserves the filter's order and therefore
avoids the change in rounding from SIMD horizontal reductions. This fork's
native_fma.py already chooses hardware FMA rather than emulated FPCR/FPSR.
"""
from __future__ import annotations

FUNCTION = 0x8338EFE0
HOOK_AFTER = 0x8338F028  # MOV W12,W10; immediately before the 1025-tap loop
POST_LOOP = 0x8338F058  # CMP W10,#0x3ff; first original post-filter step

# Exact instruction words from the supported image; fail closed on other
# revisions rather than patching a coincidentally numbered function.
WORDS = {
    0x8338F020: 0x2F00E400,  # MOVI D0,#0 (zero accumulator)
    0x8338F024: 0xB960080A,  # LDR W10,[X0,#0x2008]
    0x8338F028: 0x2A0A03EC,  # MOV W12,W10
    0x8338F02C: 0x8B2C492D,  # ADD X13,X9,W12,UXTW #2
    0x8338F034: 0xBC687801,  # LDR S1,[X0,X8,LSL #2] (coefficients)
    0x8338F044: 0xBD4001A2,  # LDR S2,[X13] (ring)
    0x8338F04C: 0x1F010040,  # FMADD S0,S2,S1,S0
    0x8338F054: 0x54FFFEC1,  # B.NE 0x8338f02c
    0x8338F058: 0x710FFD5F,  # CMP W10,#0x3ff
}

# Install after the MOV W12,W10 instruction; the original label is also
# installed in the lifter so a local goto is legal in the generated C.
#
# A64_PTR uses the guest's fastmem arena. All reads are inside the same guest
# arrays the ARM loop reads. The ring index is 0..1024 during normal operation;
# an unexpected value deliberately uses the original lifted slow path.
# A64_POLL keeps the source lift's cooperative polling path reachable, but
# checks every 64 taps instead of every backedge.
HOOK = r"""
    /* nx2/native_fir.py: f_8338efe0's 1025-tap FIR, native x86 FMA */
    if (__builtin_expect((uint32_t)c->x[12] <= 1024u, 1)) {
        const uint64_t _fir_coeff = c->x[0];
        const uint64_t _fir_ring = c->x[9];
        const float *_fir_h = (const float *)A64_PTR(_fir_coeff);
        const float *_fir_x = (const float *)A64_PTR(_fir_ring);
        const uint32_t _fir_start = (uint32_t)c->x[12];
        uint32_t _fir_idx = _fir_start;
        float _fir_sum = 0.0f;
        for (uint32_t _fir_k = 0; _fir_k < 1025u; ++_fir_k) {
            _fir_sum = __builtin_fmaf(_fir_x[_fir_idx], _fir_h[_fir_k], _fir_sum);
            _fir_idx = _fir_idx ? _fir_idx - 1u : 1024u;
            if ((_fir_k & 63u) == 63u) A64_POLL(c);
        }
        uint32_t _fir_bits;
        memcpy(&_fir_bits, &_fir_sum, sizeof(_fir_bits));
        a64_vset64(c->vreg[0], (uint64_t)_fir_bits);
        c->x[8] = 1025u;
        c->x[12] = _fir_idx;
        c->x[13] = _fir_ring + ((uint64_t)((_fir_start + 1u) % 1025u) << 2);
        goto L_8338f058;
    }
"""

# f_8338d510 contains two additional 513-tap FIR paths. The computation is
# identical; only its output scaling (S9 vs S8) differs after the loop.
FIR513_FUNCTION = 0x8338D510
FIR513_SITES = (
    (0x8338D904, 0x8338D934),
    (0x8338DE58, 0x8338DE88),
)
FIR513_WORDS = {
    0x8338D650: 0x52804017,  # MOV W23,#512: first path ring wrap
    0x8338DD90: 0x5280401A,  # MOV W26,#512: second path ring wrap
    0x8338D8DC: 0xAA1F03E9,  # MOV X9,XZR
    0x8338D8E8: 0x9120110A,  # ADD X10,X8,#0x804
    0x8338D8FC: 0x2F00E400,  # MOVI D0,#0
    0x8338D900: 0xB950090B,  # LDR W11,[X8,#0x1008]
    0x8338D904: 0x2A0B03EC,  # MOV W12,W11
    0x8338D908: 0x8B2C494D,  # ADD X13,X10,W12,UXTW #2
    0x8338D90C: 0x7108059F,  # CMP W12,#513
    0x8338D910: 0xBC696901,  # LDR S1,[X8,X9]
    0x8338D914: 0x91001129,  # ADD X9,X9,#4
    0x8338D918: 0x9A8A31AD,  # CSEL X13,X13,X10,LO
    0x8338D91C: 0x7100058C,  # SUBS W12,W12,#1
    0x8338D920: 0xBD4001A2,  # LDR S2,[X13]
    0x8338D924: 0x1A8CB2EC,  # CSEL W12,W23,W12,LT
    0x8338D928: 0x1F010040,  # FMADD S0,S2,S1,S0
    0x8338D92C: 0xF120113F,  # CMP X9,#0x804
    0x8338D930: 0x54FFFEC1,  # B.NE 0x8338d908
    0x8338D934: 0x1E290800,  # FMUL S0,S0,S9
    0x8338DE30: 0xAA1F03E9,  # MOV X9,XZR
    0x8338DE3C: 0x9120110A,  # ADD X10,X8,#0x804
    0x8338DE50: 0x2F00E400,  # MOVI D0,#0
    0x8338DE54: 0xB950090B,  # LDR W11,[X8,#0x1008]
    0x8338DE58: 0x2A0B03EC,  # MOV W12,W11
    0x8338DE5C: 0x8B2C494D,  # ADD X13,X10,W12,UXTW #2
    0x8338DE60: 0x7108059F,  # CMP W12,#513
    0x8338DE64: 0xBC696901,  # LDR S1,[X8,X9]
    0x8338DE68: 0x91001129,  # ADD X9,X9,#4
    0x8338DE6C: 0x9A8A31AD,  # CSEL X13,X13,X10,LO
    0x8338DE70: 0x7100058C,  # SUBS W12,W12,#1
    0x8338DE74: 0xBD4001A2,  # LDR S2,[X13]
    0x8338DE78: 0x1A8CB34C,  # CSEL W12,W26,W12,LT
    0x8338DE7C: 0x1F010040,  # FMADD S0,S2,S1,S0
    0x8338DE80: 0xF120113F,  # CMP X9,#0x804
    0x8338DE84: 0x54FFFEC1,  # B.NE 0x8338de5c
    0x8338DE88: 0x1E280800,  # FMUL S0,S0,S8
}

FIR513_HOOK_TEMPLATE = r"""
    /* f_8338d510: 513-tap scalar FIR, keeping ARM FMADD order. */
    if (__builtin_expect((uint32_t)c->x[12] < 513u, 1)) {
        const uint64_t _fir513_coeff_base = c->x[8];
        const uint64_t _fir513_ring_base = c->x[10];
        const float *_fir513_h = (const float *)A64_PTR(_fir513_coeff_base);
        const float *_fir513_x = (const float *)A64_PTR(_fir513_ring_base);
        const uint32_t _fir513_start = (uint32_t)c->x[12];
        uint32_t _fir513_index = _fir513_start;
        float _fir513_sum = 0.0f;
        for (uint32_t _fir513_k = 0; _fir513_k < 513u; ++_fir513_k) {
            const float _fir513_coeff = _fir513_h[_fir513_k];
            const float _fir513_sample = _fir513_x[_fir513_index];
            _fir513_sum = __builtin_fmaf(_fir513_sample, _fir513_coeff, _fir513_sum);
            _fir513_index = _fir513_index ? _fir513_index - 1u : 512u;
            if ((_fir513_k & 63u) == 63u) A64_POLL(c);
        }
        /* The final LDR S1/S2 writes zero into their upper SIMD lanes. */
        const uint32_t _fir513_last =
            _fir513_start == 512u ? 0u : _fir513_start + 1u;
        uint32_t _fir513_sum_bits, _fir513_h_bits, _fir513_x_bits;
        memcpy(&_fir513_sum_bits, &_fir513_sum, sizeof(_fir513_sum_bits));
        memcpy(&_fir513_h_bits, &_fir513_h[512], sizeof(_fir513_h_bits));
        memcpy(&_fir513_x_bits, &_fir513_x[_fir513_last], sizeof(_fir513_x_bits));
        a64_vset64(c->vreg[0], (uint64_t)_fir513_sum_bits);
        a64_vset64(c->vreg[1], (uint64_t)_fir513_h_bits);
        a64_vset64(c->vreg[2], (uint64_t)_fir513_x_bits);
        c->x[9] = 0x804u;
        c->x[12] = _fir513_index;
        c->x[13] = _fir513_ring_base + ((uint64_t)_fir513_last << 2);
        c->n = 0; c->z = 1; c->c = 1; c->v = 0; /* CMP X9,#0x804 */
        goto L_@END@;
    }
"""


def fir513_hook(end):
    """Generate the same native dot product with the correct resume address."""
    return FIR513_HOOK_TEMPLATE.replace("@END@", "%08x" % end)


def install(lifter, img, build_id=None):
    """Install matching 1025-tap and 513-tap loops for this verified build."""
    import oot_patches

    if build_id != oot_patches.LIFE_BUILD_ID:
        return False

    use1025 = FUNCTION in lifter.funcs
    use513 = FIR513_FUNCTION in lifter.funcs
    if not (use1025 or use513):
        return False

    # Validate every active hook before changing lifter state. Refuse to
    # partially patch a different build or any unexpectedly edited function.
    checks = []
    if use1025:
        checks.append((FUNCTION, WORDS, ((HOOK_AFTER, POST_LOOP),)))
    if use513:
        checks.append((FIR513_FUNCTION, FIR513_WORDS, FIR513_SITES))

    for func, words, sites in checks:
        body, _ = lifter.funcs[func]
        if any(pc not in body or body[pc][1] != word or img.word(pc) != word
               for pc, word in words.items()):
            raise RuntimeError("native_fir: unexpected instruction in %08x" % func)
        for after, resume in sites:
            if (body[after][0][0] != "op" or resume not in body
                    or after in lifter.after):
                raise RuntimeError("native_fir: invalid/occupied loop at %08x" % after)

    if use1025:
        _, labels = lifter.funcs[FUNCTION]
        lifter.no_local.add(FUNCTION)
        labels.add(POST_LOOP)
        lifter.after[HOOK_AFTER] = HOOK
        print("native FIR: hooked f_8338efe0 1025-tap loop", flush=True)

    if use513:
        _, labels = lifter.funcs[FIR513_FUNCTION]
        # Hooks update the guest register file directly; disabling per-function
        # register promotion keeps lifted code and the new path consistent.
        lifter.no_local.add(FIR513_FUNCTION)
        for after, resume in FIR513_SITES:
            labels.add(resume)
            lifter.after[after] = fir513_hook(resume)
        print("native FIR: hooked f_8338d510 two 513-tap loops", flush=True)

    return True
