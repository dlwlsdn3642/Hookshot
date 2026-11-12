"""Experimental native-only FMA for lifted AArch64 FMADD/FMLA helpers.

Replace only the generated fma32/fma32 and fma64/fma64 helper bodies, including
their hot-function copies. No FPCR/FPSR checks or soft-float fallback. Keep
third_party/a64-kit untouched so upstream updates are easy to incorporate.
"""
 
NATIVE_HELPERS = r"""
/* Fork experiment: native FMA only. ARM FPCR/FPSR and exceptional-value
   compatibility are intentionally not emulated by these two helpers. */
#if !defined(__FMA__)
#error "Native FMA experiment requires an FMA-capable x86-64 target (e.g. -march=x86-64-v3)"
#endif
static inline __attribute__((always_inline))
uint64_t a64_native_fma32(uint64_t a0, uint64_t a1, uint64_t a2) {
    uint32_t x = (uint32_t)a0, y = (uint32_t)a1, z = (uint32_t)a2, bits;
    float fx, fy, fz, result;
    memcpy(&fx, &x, sizeof(fx));
    memcpy(&fy, &y, sizeof(fy));
    memcpy(&fz, &z, sizeof(fz));
    result = __builtin_fmaf(fx, fy, fz);
    memcpy(&bits, &result, sizeof(bits));
    return (uint64_t)bits;
}
static inline __attribute__((always_inline))
uint64_t a64_native_fma64(uint64_t a0, uint64_t a1, uint64_t a2) {
    double x, y, z, result;
    uint64_t bits;
    memcpy(&x, &a0, sizeof(x));
    memcpy(&y, &a1, sizeof(y));
    memcpy(&z, &a2, sizeof(z));
    result = __builtin_fma(x, y, z);
    memcpy(&bits, &result, sizeof(bits));
    return bits;
}
"""


def patch_header(header, used):
    """Replace the two generated FMA guard/fallback shapes; fail on layout drift."""
    lines = header.splitlines(keepends=True)
    changed = False
    args = "Ctx *c, uint64_t a0, uint64_t a1, uint64_t a2"
    for bits in (32, 64):
        shape = ("fma%d" % bits, "fma%d" % bits, 3, 3)
        if shape not in used:
            continue
        changed = True
        name = "a64_fpx_fma%d_fma%d" % (bits, bits)
        hot_name = "a64_fpx_hot_fma%d_fma%d" % (bits, bits)
        for prefix, replacement in (
            ("A64_FPX_FN uint64_t %s(%s) {" % (name, args),
             "A64_FPX_FN uint64_t %s(%s) { (void)c; return a64_native_fma%d(a0, a1, a2); }\n"
             % (name, args, bits)),
            ("static inline __attribute__((always_inline)) uint64_t %s(%s) {" % (hot_name, args),
             "static inline __attribute__((always_inline)) uint64_t %s(%s) { (void)c; return a64_native_fma%d(a0, a1, a2); }\n"
             % (hot_name, args, bits)),
        ):
            positions = [i for i, line in enumerate(lines) if line.startswith(prefix)]
            if len(positions) != 1:
                raise RuntimeError("native_fma: expected one generated definition of %s, found %d" %
                                   (prefix, len(positions)))
            lines[positions[0]] = replacement
    return NATIVE_HELPERS + "".join(lines) if changed else header


_installed = False


def install():
    """Hook the normal a64_fpx.h generator, without editing vendored sources."""
    global _installed
    if _installed:
        return
    import a64_lift

    original_fpx_header = a64_lift.fpx_header

    def native_fpx_header(used):
        return patch_header(original_fpx_header(used), used)

    a64_lift.fpx_header = native_fpx_header
    _installed = True
