"""The ratio model: a scorer plus the input convention it was trained under.

The historical arrangement kept standardization statistics in a plain ``cfg["stats"]``
dict that travelled next to the weights in a JSON summary, and every consumer re-applied
them by hand. That dict was hand-reimplemented in several downstream scripts, which is the
tell: a convention that every consumer must reconstruct is a convention that will
eventually be reconstructed wrong, silently, because a mis-standardized input has the
right shape and the wrong meaning.

Here the affine standardization is part of the model -- :class:`Standardizer` buffers that
serialize with the weights and are applied inside :meth:`RatioModel.logit`. There is
nothing left for a consumer to reapply.

The split of responsibilities is deliberate:

* the EXPERIMENT owns the nonlinear feature engineering (``arcsinh(dt/20)``, ``log r``,
  ``sin/cos psi``) -- it is physics, it belongs to the pool reader;
* the MODEL owns the affine standardization of those engineered features -- it is a
  property of the training set, and pairing weights with the wrong one is silent.

:class:`RatioModel.logit` returns the classifier logit, which at the Bayes-optimal
classifier IS ``log r(x | theta) = log p(x | theta) / p(x)``. The linear head makes that
exact rather than approximate (no post-hoc sigmoid inversion).
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hitman.nn.mlp import MLP


class Standardizer(eqx.Module):
    """Frozen affine input normalization: ``(v - mean) / scale``.

    The buffers are real array leaves so they serialize with the model, but
    :meth:`__call__` wraps them in ``stop_gradient``, so they receive exactly zero
    gradient and Adam's update for them is identically zero. They are therefore
    non-trainable in effect without needing a separate filter spec that every caller would
    have to remember to pass -- one of those "remember to" requirements being precisely
    what this class exists to remove.
    """

    mean: jnp.ndarray
    scale: jnp.ndarray
    names: tuple = eqx.field(static=True, default=())

    def __init__(self, mean, scale, names=()):
        self.mean = jnp.asarray(mean, jnp.float32)
        self.scale = jnp.asarray(scale, jnp.float32)
        self.names = tuple(names)
        if self.mean.shape != self.scale.shape:
            raise ValueError(
                f"mean {self.mean.shape} and scale {self.scale.shape} must match")
        if self.names and len(self.names) != self.mean.shape[0]:
            raise ValueError(
                f"{len(self.names)} names for {self.mean.shape[0]} features")

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        return (v - jax.lax.stop_gradient(self.mean)) / jax.lax.stop_gradient(self.scale)

    @classmethod
    def from_data(cls, values, names=(), *, floor: float = 1e-6) -> "Standardizer":
        """Fit mean/scale on an array of rows. ``floor`` keeps a constant column finite."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 2:
            raise ValueError(f"expected (n_rows, n_features), got {v.shape}")
        if v.shape[0] == 0:
            raise ValueError("cannot fit a Standardizer on zero rows")
        return cls(mean=v.mean(axis=0), scale=v.std(axis=0) + floor, names=names)

    def describe(self) -> dict:
        return {"names": list(self.names),
                "mean": np.asarray(self.mean, np.float64).tolist(),
                "scale": np.asarray(self.scale, np.float64).tolist()}


class RatioModel(eqx.Module):
    """``log r(x | theta)`` -- a scorer with its input convention attached.

    ``encoder`` is an intentionally empty SLOT. Today the model scores one observation
    unit at a time and an event's log-ratio is the sum over its rows, which assumes the
    rows are conditionally independent given theta and discards multiplicity. A
    set-valued encoder (permutation-invariant over a group's rows) and a count factor are
    the known upgrades; leaving the slot here means that becomes a head swap rather than a
    data-layout rewrite. It is NOT implemented -- ``encoder`` must be ``None``.
    """

    net: MLP
    x_std: Standardizer
    theta_std: Standardizer
    encoder: Optional[eqx.Module] = None

    def __init__(self, *, x_std: Standardizer, theta_std: Standardizer, key,
                 width: int = 64, depth: int = 3, activation: str = "mish",
                 encoder=None):
        if encoder is not None:
            raise NotImplementedError(
                "set-valued encoders are a reserved slot, not yet implemented: the "
                "per-row scorer plus a sum over the group is the current likelihood. "
                "Adding an encoder changes what the objective MEANS, so it needs the "
                "validation battery (hitman.validate) standing first.")
        self.x_std = x_std
        self.theta_std = theta_std
        self.encoder = None
        self.net = MLP(in_size=x_std.dim + theta_std.dim, width=width, depth=depth,
                       key=key, activation=activation)

    @property
    def x_dim(self) -> int:
        return self.x_std.dim

    @property
    def theta_dim(self) -> int:
        return self.theta_std.dim

    @property
    def feat_dim(self) -> int:
        return self.x_dim + self.theta_dim

    def logit(self, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """ONE row's classifier logit = ``log r(x | theta)``. Inputs are UNstandardized."""
        return self.net(jnp.concatenate([self.x_std(x), self.theta_std(theta)]))

    def __call__(self, x, theta):
        return self.logit(x, theta)

    def logit_batch(self, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Vectorized over rows: ``(n, Dx), (n, Dt) -> (n,)``.

        Standardization and the concatenate are done ONCE on the whole batch, and only the
        network is vmapped. Writing this as ``vmap(self.logit)`` instead -- which is the
        obvious spelling -- pushes a per-row concatenate inside the vmap and measured
        ~1.9x slower per training step at batch 262144, because the batch here is huge and
        the network tiny, so anything per-row that is not a GEMM dominates.

        The affine standardization broadcasts over the leading axis for free, so this is
        arithmetically identical to the per-row form.
        """
        feats = jnp.concatenate([self.x_std(x), self.theta_std(theta)], axis=-1)
        return jax.vmap(self.net)(feats)

    def standardized(self, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """The concatenated standardized feature vector -- what the support guard sees."""
        return jnp.concatenate([self.x_std(x), self.theta_std(theta)])

    @property
    def feature_names(self) -> tuple:
        return tuple(self.x_std.names) + tuple(self.theta_std.names)


def build_ratio_model(dataset, *, key, width: int = 64, depth: int = 3,
                      activation: str = "mish") -> RatioModel:
    """Fit the standardizers on the dataset's TRAIN split and build the model.

    Train-split-only, always: fitting normalization on val or test rows leaks their
    distribution into the model's input convention, which is a small leak that is
    nonetheless impossible to detect after the fact.
    """
    tr = dataset.split_view("train")
    if tr.n_rows == 0:
        raise ValueError("cannot build a model: the train split is empty")
    return RatioModel(
        x_std=Standardizer.from_data(tr.x, dataset.x_names),
        theta_std=Standardizer.from_data(tr.theta, dataset.theta_names),
        key=key, width=width, depth=depth, activation=activation,
    )


def n_parameters(model: RatioModel) -> int:
    """Trainable leaf count. The standardizer buffers are counted as what they are --
    array leaves -- but receive zero gradient; see :class:`Standardizer`."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model.net, eqx.is_inexact_array))
    return int(sum(x.size for x in leaves))
