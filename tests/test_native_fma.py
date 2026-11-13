"""The fork must be able to import native_fma and emit compilable native FMA.

No game dump is required; this guards against syntax mistakes in the lifted
header generation, and prevents a silent fallback to ARM software FMA.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))

import kit  # noqa: F401,E402 (a64-kit path)
import a64_lift  # noqa: E402
import native_fma  # noqa: E402


def check():
    shapes = {("fma32", "fma32", 3, 3), ("fma64", "fma64", 3, 3)}
    native_fma.install()
    native_fma.install()  # idempotent
    assert a64_lift.fpx_header is native_fma.fpx_header
    generated = a64_lift.fpx_header(shapes)
    original = native_fma._ORIGINAL(shapes)
    assert generated != original
    assert '#if !defined(__FMA__)' in generated
    assert '__builtin_fmaf' in generated
    assert '__builtin_fma(' in generated
    for width in (32, 64):
        for hot in (False, True):
            signature, replacement = native_fma._native_definition(width, hot)
            assert generated.count(signature + " {") == 1
            assert replacement in generated
            assert "RECOMP_FPX_OPEN" not in replacement
            assert "recomp_fpx_exact" not in replacement

    # Ensure that no-op transformations leave other FPX code untouched.
    assert native_fma.fpx_header(set()) == native_fma._ORIGINAL(set())

    cc = shutil.which("gcc") or shutil.which("clang")
    if cc:
        definitions = [native_fma._native_definition(w, hot)[1]
                       for w in (32, 64) for hot in (False, True)]
        stub = ("#include <stdint.h>\n#include <string.h>\n"
                "#define A64_FPX_FN static __attribute__((noinline))\n"
                "typedef struct { int dummy; } Ctx;\n")
        code = stub + "\n".join(definitions) + "\n"
        code += ("uint64_t test32(Ctx *c, uint64_t a, uint64_t b, uint64_t z) "
                 "{ return a64_fpx_fma32_fma32(c,a,b,z); }\n")
        code += ("uint64_t test64(Ctx *c, uint64_t a, uint64_t b, uint64_t z) "
                 "{ return a64_fpx_fma64_fma64(c,a,b,z); }\n")
        with tempfile.TemporaryDirectory(prefix="hookshot-native-fma-") as td:
            src = Path(td) / "native_fma.c"
            obj = Path(td) / "native_fma.o"
            src.write_text(code, encoding="utf-8")
            p = subprocess.run([cc, "-O2", "-mfma", "-c", str(src), "-o", str(obj)],
                               capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError("FMA definitions do not compile: " + p.stderr[:1500])
    print("PASS: native scalar FP32/FP64 FMA generator, hot/normal definitions, C compile")


if __name__ == "__main__":
    check()
