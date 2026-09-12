"""Deprecation shims for the pre-split import paths.

The water-Cherenkov application moved from the top level into :mod:`hitman.wc` so that
``import hitman`` no longer implies a PMT detector. Every old path still resolves, once,
with a ``DeprecationWarning`` naming its replacement.

Mechanism: :func:`alias_module` rebinds ``sys.modules[old_name]`` to the already-imported
new module object, so the old path is the SAME object -- not a copy, not a re-export. The
import machinery re-reads ``sys.modules[spec.name]`` after executing a module and honours
the replacement, which is what makes this exact rather than approximate.

Known limitation, deliberately accepted: for a shimmed PACKAGE the alias makes
``hitman.inference`` and ``hitman.wc.inference`` the same object, but a subsequent
``import hitman.inference.mle`` resolves through the aliased package's ``__path__`` and
would create a SECOND module object for ``hitman.wc.inference.mle``. Mixing old and new
paths for the same submodule in one process is therefore not supported. Nothing in this
repo or in the muonsync pipeline does that (the pipeline imports none of the moved
modules; tests and scripts were rewritten to the new paths), and per-submodule shims for
the whole WC surface would be a large, permanently-maintained file set for a case that
does not arise. Use the new paths.
"""

import importlib
import sys
import warnings

#: old dotted path -> new dotted path
MOVED = {
    "hitman.inference": "hitman.wc.inference",
    "hitman.splinemle": "hitman.wc.splinemle",
    "hitman.likelihood": "hitman.wc.likelihood",
    "hitman.toys": "hitman.wc.toys",
    "hitman.nn.hitnet": "hitman.wc.nn.hitnet",
    "hitman.nn.chargenet": "hitman.wc.nn.chargenet",
    "hitman.nn.features": "hitman.wc.nn.features",
    "hitman.nn.frame": "hitman.wc.nn.frame",
    "hitman.data.ratds": "hitman.wc.data.ratds",
    "hitman.data.store": "hitman.wc.data.store",
    "hitman.data.sampling": "hitman.wc.data.sampling",
    "hitman.calibrate.normalization": "hitman.wc.calibrate.normalization",
    "hitman.receipts.forward": "hitman.wc.receipts.forward",
    "hitman.receipts.runner": "hitman.wc.receipts.runner",
    "hitman.receipts.harness": "hitman.wc.receipts.harness",
    "hitman.receipts.mle": "hitman.wc.receipts.mle",
    "hitman.train.event": "hitman.wc.train.event",
    "hitman.train.identities": "hitman.wc.train.identities",
    "hitman.train.moments": "hitman.wc.train.moments",
    "hitman.train.nwj": "hitman.wc.train.nwj",
    "hitman.train.polish": "hitman.wc.train.polish",
    "hitman.train.recipe": "hitman.wc.train.recipe",
    "hitman.train.reweight": "hitman.wc.train.reweight",
    "hitman.train.resident": "hitman.wc.train.resident",
    "hitman.npe.batch": "hitman.wc.npe.batch",
    "hitman.npe.train": "hitman.wc.npe.train",
}


def alias_module(old_name: str) -> None:
    """Make ``old_name`` resolve to its ``MOVED`` target, with a DeprecationWarning."""
    new_name = MOVED[old_name]
    warnings.warn(
        f"{old_name} moved to {new_name} (the water-Cherenkov application was split out "
        f"of the library). The old path still works; update the import.",
        DeprecationWarning, stacklevel=3,
    )
    sys.modules[old_name] = importlib.import_module(new_name)


def deprecated_getattr(module_name: str, names: dict):
    """Build a module-level ``__getattr__`` serving relocated attributes lazily.

    ``names`` maps attribute -> new dotted module path. Lazy so that merely importing the
    library never pulls in the WC application (that being the point of the split).
    """
    def __getattr__(name):
        target = names.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        warnings.warn(
            f"{module_name}.{name} moved to {target}.{name}; import it from there.",
            DeprecationWarning, stacklevel=2,
        )
        return getattr(importlib.import_module(target), name)
    return __getattr__
