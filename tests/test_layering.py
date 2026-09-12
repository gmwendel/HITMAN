"""The split's structural invariant: the library must not depend on the application.

This is the test that keeps the split real. Without it the boundary erodes one convenient
import at a time -- which is exactly what had already happened twice by the time the split
landed (hitman.npe.encoder reaching for the WC feature scales, and hitman.npe.batch
importing hitman.wc.train.identities), in both cases invisibly, because everything still
worked.
"""

import subprocess
import sys

LIBRARY_MODULES = [
    "hitman", "hitman.spec", "hitman.data", "hitman.nn", "hitman.density",
    "hitman.diagnostics", "hitman.receipts", "hitman.calibrate", "hitman.train",
    "hitman.npe", "hitman.ratio", "hitman.validate",
]


def test_importing_the_library_pulls_in_no_wc_modules():
    """Import-time edge check, in a SUBPROCESS so a module another test already imported
    cannot mask the dependency."""
    code = (
        "import sys\n"
        "import warnings; warnings.simplefilter('ignore')\n"
        + "".join(f"import {m}\n" for m in LIBRARY_MODULES)
        + "wc = sorted(m for m in sys.modules if m.startswith('hitman.wc'))\n"
        "print('|'.join(wc))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", (
        f"the library imported these application modules: {out.split('|')}. "
        f"hitman/ must never import from hitman/wc/ -- move the shared piece down into "
        f"the library, or move the consumer up into the application.")


def test_library_source_has_no_wc_import_statements():
    """Static check, complementing the runtime one: a lazily-imported edge inside a
    function would pass the test above but is still a dependency."""
    import ast
    import pathlib

    offenders = []
    root = pathlib.Path(__file__).resolve().parent.parent / "hitman"
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] == "wc" or rel.name == "_compat.py":
            continue          # the application may import the library; _compat maps names
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("hitman.wc"):
                        offenders.append(f"{rel}:{node.lineno} import {a.name}")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hitman.wc"):
                offenders.append(f"{rel}:{node.lineno} from {node.module} import ...")
    assert not offenders, "library modules importing the application:\n  " + "\n  ".join(offenders)


def test_deprecated_paths_still_resolve_to_the_same_object():
    """The shims must be aliases, not copies -- a copy would break isinstance across paths."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import hitman.wc.nn.hitnet as new
        import hitman.nn.hitnet as old
        import hitman.wc.npe.batch as new_batch
        import hitman.npe.batch as old_batch
    assert old is new
    assert old_batch is new_batch
    assert old.HitNet is new.HitNet
