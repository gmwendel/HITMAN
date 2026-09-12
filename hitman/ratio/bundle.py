"""Self-describing model bundles: weights plus everything needed to use them correctly.

The failure this exists to make impossible: a checkpoint was previously reassembled by
hand from ``checkpoint.eqx`` plus three unrelated corners of a ``summary.json``
(``nre_config`` for the featurization, ``args`` for the architecture, ``feature_support``
for the guard). Any consumer that reconstructed a *different* convention from the one the
model was trained under got a model that ran, produced finite numbers, and meant something
else. That is not a hypothetical -- a checkpoint trained with one observable
parameterization would load and evaluate happily against another, because both had the
same feature width and the statistics dict carried entries for both.

A bundle is a directory:

    weights.eqx      the model pytree leaves (equinox)
    bundle.json      architecture, specs, standardizers, support, provenance
    MANIFEST         sha256 of the two above, so a partial/edited bundle is detectable

and :func:`load_bundle` REFUSES on any mismatch rather than reconstructing a plausible
model. The refusal is the feature.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import numpy as np

from hitman.ratio.model import RatioModel, Standardizer
from hitman.ratio.support import FeatureSupport

BUNDLE_SCHEMA = "hitman-ratio-bundle-v1"


@dataclass
class RatioBundle:
    """A trained ratio model with its input convention, guard and provenance."""

    model: RatioModel
    support: Optional[FeatureSupport] = None
    provenance: dict = field(default_factory=dict)
    architecture: dict = field(default_factory=dict)
    train_receipt: dict = field(default_factory=dict)

    # -- the compatibility contract -----------------------------------------
    @property
    def x_names(self) -> tuple:
        return tuple(self.model.x_std.names)

    @property
    def theta_names(self) -> tuple:
        return tuple(self.model.theta_std.names)

    def assert_compatible(self, dataset_or_names, *, what: str = "dataset") -> None:
        """Refuse a dataset whose feature layout differs from the training layout.

        Checks NAMES and ORDER, not just widths. Order matters because every consumer
        indexes positionally; a permutation of the same names is exactly the kind of
        mismatch that produces finite, wrong numbers.
        """
        if hasattr(dataset_or_names, "x_names"):
            x_names = tuple(dataset_or_names.x_names)
            th_names = tuple(dataset_or_names.theta_names)
        else:
            x_names, th_names = (tuple(n) for n in dataset_or_names)
        if not x_names or not th_names:
            raise ValueError(
                f"{what} carries no feature names, so compatibility cannot be checked. "
                f"Unnamed features are how a mismatch stays silent -- name them.")
        if x_names != self.x_names or th_names != self.theta_names:
            raise ValueError(
                f"feature layout mismatch between the bundle and this {what}:\n"
                f"  bundle  x={self.x_names}\n          theta={self.theta_names}\n"
                f"  {what:<7} x={x_names}\n          theta={th_names}\n"
                f"Same width is not the same meaning; refusing rather than evaluating a "
                f"model on features it was not trained on.")

    def require_support(self) -> FeatureSupport:
        """The recorded training support, or a loud failure.

        A bundle with no support cannot be used for a guarded fit, and a guarded fit is
        the default. Callers that genuinely want an unguarded objective say so explicitly
        rather than falling into it because a field happened to be absent.
        """
        if self.support is None:
            raise ValueError(
                "this bundle records no feature support, so no out-of-distribution guard "
                "can be built. Recompute it with hitman.ratio.support.compute_support and "
                "re-save, or pass guard=None to accept an unguarded objective knowingly.")
        return self.support

    def describe(self) -> dict:
        return {
            "schema": BUNDLE_SCHEMA,
            "architecture": self.architecture,
            "x_names": list(self.x_names), "theta_names": list(self.theta_names),
            "x_dim": self.model.x_dim, "theta_dim": self.model.theta_dim,
            "has_support": self.support is not None,
            "provenance": self.provenance,
            "train_receipt": self.train_receipt,
        }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(path: str, bundle: RatioBundle) -> str:
    """Write a bundle atomically (tmp dir + ``os.replace``)."""
    m = bundle.model
    meta = {
        "schema": BUNDLE_SCHEMA,
        "architecture": {
            "width": int(m.net.layers[0].out_features),
            "depth": len(m.net.layers) - 1,
            "activation": m.net.act,
            "in_size": int(m.net.layers[0].in_features),
            **(bundle.architecture or {}),
        },
        "x_std": m.x_std.describe(),
        "theta_std": m.theta_std.describe(),
        "support": None if bundle.support is None else bundle.support.to_dict(),
        "provenance": bundle.provenance,
        "train_receipt": bundle.train_receipt,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(tmp, "weights.eqx"), m)
    with open(os.path.join(tmp, "bundle.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(os.path.join(tmp, "MANIFEST"), "w") as f:
        json.dump({"schema": BUNDLE_SCHEMA,
                   "weights.eqx": _sha256(os.path.join(tmp, "weights.eqx")),
                   "bundle.json": _sha256(os.path.join(tmp, "bundle.json"))}, f, indent=2)
    if os.path.exists(path):
        stale = f"{path}.old.{os.getpid()}"
        os.replace(path, stale)
        shutil.rmtree(stale, ignore_errors=True)
    os.replace(tmp, path)
    return path


def load_bundle(path: str, *, verify: bool = True) -> RatioBundle:
    """Read a bundle, rebuilding the model from its OWN recorded architecture.

    Nothing is inferred and nothing is defaulted: the skeleton comes from the bundle's
    architecture block and the standardizers from its own buffers, so a loaded model is
    the model that was saved or the load fails.
    """
    meta_path = os.path.join(path, "bundle.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{path!r} is not a ratio bundle (no bundle.json)")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(
            f"{path!r}: bundle schema {meta.get('schema')!r}, expected {BUNDLE_SCHEMA!r}. "
            f"Refusing to guess at an older layout.")
    if verify:
        man_path = os.path.join(path, "MANIFEST")
        if not os.path.isfile(man_path):
            raise ValueError(f"{path!r}: MANIFEST missing -- bundle integrity unverifiable")
        with open(man_path) as f:
            man = json.load(f)
        for name in ("weights.eqx", "bundle.json"):
            got = _sha256(os.path.join(path, name))
            if got != man.get(name):
                raise ValueError(
                    f"{path!r}: {name} sha256 {got[:12]} != manifest {str(man.get(name))[:12]}"
                    f" -- the bundle was edited or is partially written.")

    arch = meta["architecture"]
    x_std = Standardizer(np.asarray(meta["x_std"]["mean"]),
                         np.asarray(meta["x_std"]["scale"]),
                         tuple(meta["x_std"]["names"]))
    th_std = Standardizer(np.asarray(meta["theta_std"]["mean"]),
                          np.asarray(meta["theta_std"]["scale"]),
                          tuple(meta["theta_std"]["names"]))
    skeleton = RatioModel(x_std=x_std, theta_std=th_std, key=jax.random.PRNGKey(0),
                          width=int(arch["width"]), depth=int(arch["depth"]),
                          activation=arch["activation"])
    model = eqx.tree_deserialise_leaves(os.path.join(path, "weights.eqx"), skeleton)
    support = (FeatureSupport.from_dict(meta["support"])
               if meta.get("support") is not None else None)
    return RatioBundle(model=model, support=support,
                       provenance=meta.get("provenance", {}),
                       architecture=arch,
                       train_receipt=meta.get("train_receipt", {}))
