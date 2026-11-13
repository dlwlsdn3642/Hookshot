"""Reduce the OoT microphone's 1025-tap FIR loop to native scalar FMA.

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


def install(lifter, img, build_id=None):
    """Apply to a real lift, only if the complete guest instruction signature matches."""
    # The build check is redundant with the words, but protects a future
    # game image with identical byte patterns at accidentally reused addresses.
    import oot_patches
    if build_id != oot_patches.LIFE_BUILD_ID:
        return False
    if FUNCTION not in lifter.funcs:
        return False
    body, labels = lifter.funcs[FUNCTION]
    if (any(pc not in body or body[pc][1] != word for pc, word in WORDS.items())
            or any(img.word(pc) != word for pc, word in WORDS.items())
            or body[HOOK_AFTER][0][0] != "op" or POST_LOOP not in body):
        raise RuntimeError("native_fir: supported build has an unexpected FIR loop; not patching")
    if HOOK_AFTER in lifter.after:
        raise RuntimeError("native_fir: f_8338efe0 loop hook already occupied")
    # No local-register promotion for this one guest function: the hook's
    # direct writes to Ctx must be visible to the original post-loop code.
    lifter.no_local.add(FUNCTION)
    labels.add(POST_LOOP)
    lifter.after[HOOK_AFTER] = HOOK
    print("native FIR: hooked f_8338efe0 1025-tap loop (guest %#x)" % HOOK_AFTER, flush=True)
    return True
