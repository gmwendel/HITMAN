"""Training-support recording and the out-of-distribution guard.

A ratio estimator is an estimator only inside the region it was trained on. Outside it the
network extrapolates and its logit is unconstrained fiction -- and a downstream fit that
MINIMIZES ``-sum(logit)`` will walk straight into that fiction if nothing records where
the training data actually was.

Two facts, both measured on the muon program, shape this module:

1. **A per-feature box is not enough.** The spurious solution that motivated this sat
    inside the literal per-feature train min/max on 8 of 9 features (the ninth exceeded it
    by 0.049), yet had ZERO training neighbours within 2 z-units and lay 3.5 z-units from
    its nearest of ~10^8 training rows. A box cannot express "in range on every axis, in a
    combination that never occurs".
2. **Tightening the box to catch it fires on the physics.** Tightened to the 0.1-99.9%
    bulk, the box rejected most legitimate fits, because one feature's bulk is +-8.4 while
    its true range is +-45.5.

So the guard is the JOINT Mahalanobis distance in the standardized feature space, and the
box is retained for diagnostics only. The barrier is ``relu(d_M - d0)^2``: exactly zero
inside the trust radius, so it cannot bias an in-distribution fit, and quadratic outside.

Per-anchor trust radius. ``d0`` is not a single global constant. Some legitimate
operating points are themselves out of distribution, and a guard anchored on the training
quantile alone would silently re-fit exactly those points -- changing a result rather than
protecting it. :func:`trust_radius` therefore takes the max of the training quantile and
(this anchor's own distance + a margin), so the barrier is identically zero AT the anchor
whatever the anchor is, and only bites when a fit tries to wander further out than where
it started. Tightening the quantile is then always safe.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

#: quantiles of the training Mahalanobis distance recorded for later anchoring
DM_PROBS = (0.5, 0.9, 0.99, 0.999, 0.9999)
#: default per-feature tail for the (diagnostic) box
TAIL_Q = 0.001
#: rows subsampled for quantile/covariance estimation; extrema stay exact
MAX_QUANTILE_ROWS = 20_000_000


@dataclass(frozen=True)
class FeatureSupport:
    """Where the training data was, in the model's standardized feature space.

    Standardized space matters: the covariance is estimated and the distance measured in
    the same coordinates the network sees, so the guard travels with the model rather than
    with whatever the caller happened to featurize with.
    """

    names: tuple
    lo: np.ndarray          # (D,) exact per-feature min
    hi: np.ndarray          # (D,) exact per-feature max
    q_lo: np.ndarray        # (D,) tail_q quantile
    q_hi: np.ndarray        # (D,) 1 - tail_q quantile
    mean: np.ndarray        # (D,)
    cov: np.ndarray         # (D, D)
    cov_inv: np.ndarray     # (D, D)
    dm_probs: tuple
    dm_quantiles: np.ndarray
    tail_q: float
    n_rows: int
    n_rows_estimate: int

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def d_mahalanobis(self, feat) -> np.ndarray:
        """Whitened distance from the training mean. ``feat`` is (D,) or (n, D)."""
        f = np.atleast_2d(np.asarray(feat, dtype=np.float64))
        dx = f - self.mean
        d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dx, self.cov_inv, dx), 0.0))
        return d[0] if np.ndim(feat) == 1 else d

    def dm_at(self, prob: float) -> float:
        """The recorded training-distance quantile nearest ``prob``."""
        i = int(np.argmin(np.abs(np.asarray(self.dm_probs) - prob)))
        return float(self.dm_quantiles[i])

    def in_box(self, feat) -> np.ndarray:
        f = np.atleast_2d(np.asarray(feat, dtype=np.float64))
        return np.all((f >= self.q_lo) & (f <= self.q_hi), axis=1)

    def flag(self, feat, *, prob: float = 0.999) -> dict:
        """Diagnostic verdict for one or more feature rows."""
        d = np.atleast_1d(self.d_mahalanobis(feat))
        thresh = self.dm_at(prob)
        return {
            "d_mahalanobis": d.tolist(),
            "d_max": float(d.max()) if d.size else float("nan"),
            "threshold": thresh, "threshold_prob": float(prob),
            "n_out_of_distribution": int(np.count_nonzero(d > thresh)),
            "fraction_out_of_distribution": (float(np.mean(d > thresh)) if d.size else 0.0),
            "in_box": self.in_box(feat).tolist(),
        }

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "names": list(self.names), "lo": self.lo.tolist(), "hi": self.hi.tolist(),
            "q_lo": self.q_lo.tolist(), "q_hi": self.q_hi.tolist(),
            "mean": self.mean.tolist(), "cov": self.cov.tolist(),
            "cov_inv": self.cov_inv.tolist(),
            "dm_probs": list(self.dm_probs),
            "dm_quantiles": self.dm_quantiles.tolist(),
            "tail_q": float(self.tail_q), "n_rows": int(self.n_rows),
            "n_rows_estimate": int(self.n_rows_estimate),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSupport":
        return cls(
            names=tuple(d["names"]), lo=np.asarray(d["lo"]), hi=np.asarray(d["hi"]),
            q_lo=np.asarray(d["q_lo"]), q_hi=np.asarray(d["q_hi"]),
            mean=np.asarray(d["mean"]), cov=np.asarray(d["cov"]),
            cov_inv=np.asarray(d["cov_inv"]), dm_probs=tuple(d["dm_probs"]),
            dm_quantiles=np.asarray(d["dm_quantiles"]), tail_q=float(d["tail_q"]),
            n_rows=int(d["n_rows"]), n_rows_estimate=int(d["n_rows_estimate"]),
        )


def compute_support(model, dataset, *, split: str = "train", tail_q: float = TAIL_Q,
                    max_rows: int = MAX_QUANTILE_ROWS, seed: int = 0,
                    chunk_rows: int = 2_000_000) -> FeatureSupport:
    """Record the training support of ``model``'s standardized input.

    Extrema are exact over every row; quantiles and the covariance come from a seeded
    subsample (a full sort of ~10^8 x D is pointless precision for a 0.1% tail). The
    subsample is SORTED before gathering, which matters when the arrays are memory-mapped.

    Streamed in chunks, and standardized with numpy rather than through the model's jax
    path. Standardization is affine, so the buffers can simply be pulled to host: routing
    ~10^8 rows through ``jnp.asarray`` would put the entire feature table on the device
    purely to subtract a mean, which is a device OOM at pool scale and pointless at any
    scale. Only ``max_rows`` of the standardized data is ever held at once.
    """
    sub = dataset.split_view(split) if split else dataset
    n_rows = int(sub.n_rows)
    if n_rows == 0:
        raise ValueError(f"compute_support: split {split!r} is empty")

    x_mu = np.asarray(model.x_std.mean, np.float64)
    x_sc = np.asarray(model.x_std.scale, np.float64)
    t_mu = np.asarray(model.theta_std.mean, np.float64)
    t_sc = np.asarray(model.theta_std.scale, np.float64)
    dim = x_mu.size + t_mu.size

    rng = np.random.default_rng(seed)
    if n_rows > max_rows:
        want = np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))
    else:
        want = np.arange(n_rows)

    lo = np.full(dim, np.inf)
    hi = np.full(dim, -np.inf)
    parts = []
    step = max(1, int(chunk_rows))
    for s in range(0, n_rows, step):
        e = min(s + step, n_rows)
        f = np.concatenate([
            (np.asarray(sub.x[s:e], np.float64) - x_mu) / x_sc,
            (np.asarray(sub.theta[s:e], np.float64) - t_mu) / t_sc,
        ], axis=1)
        lo = np.minimum(lo, f.min(axis=0))
        hi = np.maximum(hi, f.max(axis=0))
        j0, j1 = np.searchsorted(want, s), np.searchsorted(want, e)
        if j1 > j0:
            parts.append(f[want[j0:j1] - s])
    sample = np.concatenate(parts) if parts else np.empty((0, dim))
    q_lo, q_hi = np.quantile(sample, [tail_q, 1.0 - tail_q], axis=0)
    mean = sample.mean(axis=0)
    cov = np.cov(sample, rowvar=False)
    cov = np.atleast_2d(cov)
    # ridge: sin/cos pairs and any exactly-collinear engineered feature would otherwise
    # make this singular. Small enough not to move a well-conditioned distance.
    cov_inv = np.linalg.inv(cov + 1e-9 * np.eye(cov.shape[0]))
    dx = sample - mean
    d_m = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dx, cov_inv, dx), 0.0))
    return FeatureSupport(
        names=tuple(model.feature_names), lo=lo, hi=hi,
        q_lo=q_lo, q_hi=q_hi, mean=mean, cov=cov, cov_inv=cov_inv,
        dm_probs=DM_PROBS, dm_quantiles=np.quantile(d_m, DM_PROBS),
        tail_q=float(tail_q), n_rows=n_rows, n_rows_estimate=int(sample.shape[0]),
    )


def trust_radius(support: Optional[FeatureSupport], anchor_feat=None, *,
                 prob: float = 0.999, margin: float = 2.0) -> float:
    """The ``d0`` at which the barrier starts, anchored so it is zero AT ``anchor_feat``.

    ``max(train quantile, d_M(anchor) + margin)``. The second term is the load-bearing
    one: an operating point that is itself out of distribution must not be dragged back by
    the guard -- that would silently change a result rather than protect it. Such points
    should be REPORTED as out-of-distribution (see :meth:`FeatureSupport.flag`), not
    quietly re-fitted.
    """
    if support is None:
        return float("inf")
    d0 = support.dm_at(prob)
    if anchor_feat is not None:
        d_anchor = np.atleast_1d(support.d_mahalanobis(anchor_feat))
        if d_anchor.size:
            d0 = max(d0, float(d_anchor.max()) + float(margin))
    return float(d0)


def make_barrier(support: Optional[FeatureSupport], *, weight: float = 1.0):
    """Build ``penalty(standardized_feat, d0) -> scalar``, in logit units.

    Returns ``None`` when there is no support to guard against, so the caller's objective
    is then EXACTLY the unguarded one rather than an accidentally-zero barrier.
    """
    if support is None or float(weight) <= 0.0:
        return None
    mu = jnp.asarray(support.mean, jnp.float32)
    prec = jnp.asarray(support.cov_inv, jnp.float32)
    w = float(weight)

    def penalty(feat, d0):
        dx = feat - mu
        # +1e-12 keeps the sqrt's gradient finite exactly at the mean
        d_m = jnp.sqrt(jnp.maximum(dx @ (prec @ dx), 0.0) + 1e-12)
        return w * jnp.maximum(d_m - d0, 0.0) ** 2

    return penalty
