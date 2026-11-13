"""Fork-only FPX generator override: hardware FMA without ARM FPCR/FPSR emulation.

Only the scalar fused multiply-add helpers (32/64-bit) change. Their normal
and hot copies return the x86 FMA result directly; there is no guest-flag
gate, sticky rescue, or exact software fallback. This is intentionally
approximate for NaNs, subnormals, rounding modes, and FPSR flags.

The change lives in nx2 so upstream a64-kit stays untouched.
"""
import a64_lift


_INSTALLED = False
_ORIGINAL = a64_lift.fpx_header


def _native_definition(width, hot):
    kind = "fma%d" % width
    qualifier = ("static inline __attribute__((always_inline))"
                 if hot else "A64_FPX_FN")
    name = "a64_fpx_%s%s_%s" % ("hot_" if hot else "", kind, kind)
    signature = ("%s uint64_t %s(Ctx *c, uint64_t a0, uint64_t a1, uint64_t a2)"
                 % (qualifier, name))
    int_type = "uint32_t" if width == 32 else "uint64_t"
    real_type = "float" if width == 32 else "double"
    fused = "__builtin_fmaf" if width == 32 else "__builtin_fma"
    # memcpy bitcasts compile to register moves, not function calls.
    body = [
        "(void)c;",
        "%s ia = (%s)a0, ib = (%s)a1, iz = (%s)a2, ir;" %
        (int_type, int_type, int_type, int_type),
        "%s a, b, z, value;" % real_type,
        "memcpy(&a, &ia, sizeof(a));",
        "memcpy(&b, &ib, sizeof(b));",
        "memcpy(&z, &iz, sizeof(z));",
        "value = %s(a, b, z);" % fused,
        "memcpy(&ir, &value, sizeof(ir));",
        "return (uint64_t)ir;",
    ]
    return signature, signature + " { " + " ".join(body) + " }"


def fpx_header(used):
    text = _ORIGINAL(used)
    target_shapes = {("fma32", "fma32", 3, 3): 32,
                     ("fma64", "fma64", 3, 3): 64}
    active = [width for shape, width in target_shapes.items() if shape in used]
    if not active:
        return text

    lines = text.splitlines()
    for width in active:
        for hot in (False, True):
            signature, replacement = _native_definition(width, hot)
            matching = [i for i, line in enumerate(lines)
                        if line.startswith(signature + " {")]
            if len(matching) != 1:
                raise RuntimeError("native_fma: expected exactly one %s definition, found %d"
                                   % (signature, len(matching)))
            lines[matching[0]] = replacement

    # Native-only means a missing FMA instruction is a build error,
    # never a silent library/soft-float fallback.
    check = [
        "/* Fork native-only FMADD/FMLA: not ARM FPCR/FPSR bit-exact. */",
        "#if !defined(__FMA__)",
        '#error "Native FMA needs an FMA-capable build (RX_MARCH=x86-64-v3)."',
        "#endif",
    ]
    return "\n".join(check + lines) + "\n"


def install():
    global _INSTALLED
    if not _INSTALLED:
        a64_lift.fpx_header = fpx_header
        _INSTALLED = True
