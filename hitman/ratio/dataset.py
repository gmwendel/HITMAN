"""The ratio lane's data contract.

A :class:`RatioDataset` is a columnar view over already-featurized rows. It is
deliberately NOT a loader: the library never opens parquet, ROOT, or anything else. An
experiment's own reader produces these arrays; everything downstream (splitting, pairing,
training, reweighting, validation) keys off this one structure.

Four of the columns are the reason the class exists rather than a bare tuple of arrays:

``group_id``
    The exchangeability unit -- the thing a hypothesis belongs to (an air shower, an
    event). Splits, negative pairing and reweighting all key on it. Without it, a
    row-level split leaks (rows of one group land on both sides) and the marginal negative
    is not a marginal draw. See :mod:`hitman.ratio.pairing`.

``weight``
    Per-row SIMULATION weight (export prescale, thinning). Training on a weighted sample
    as if unweighted is a density bias that reads as a physics result rather than a bug.

``log_q``
    Log-density of the PROPOSAL that produced this row's hypothesis, broadcast from the
    group. This is what makes multi-source training and retargeting to a different prior a
    column operation instead of a re-derivation from a config file that may since have
    changed.

``source_id``
    Which pool/campaign a row came from. Needed to report effective sample size and
    calibration PER SOURCE: a set of pools that individually miscalibrate can pool to
    something that looks fine.

All four are REQUIRED. That is the design decision, not an oversight: an unweighted,
single-source dataset is constructed with :meth:`unweighted`, which fills them
explicitly. You have to say it, so you cannot forget it.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class RatioDataset:
    """Featurized rows plus the grouping/weighting/provenance columns.

    ``x`` and ``theta`` are the ENGINEERED features (arcsinh/log/sin-cos already applied)
    but NOT standardized: the affine standardization is a buffer on the model, so that a
    saved model carries its own input convention and cannot be paired with the wrong one.
    See :class:`hitman.ratio.model.Standardizer`.
    """

    x: np.ndarray            # (N, Dx) float32
    theta: np.ndarray        # (N, Dt) float32
    group_id: np.ndarray     # (N,)    int64
    weight: np.ndarray       # (N,)    float32 -- simulation weight
    log_q: np.ndarray        # (N,)    float64 -- proposal log-density of this row's theta
    source_id: np.ndarray    # (N,)    int32
    split: np.ndarray        # (N,)    '<U5' in SPLITS
    x_names: tuple = ()
    theta_names: tuple = ()

    def __post_init__(self):
        n = self.x.shape[0]
        for name in ("theta", "group_id", "weight", "log_q", "source_id", "split"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"RatioDataset column {name!r} has {arr.shape[0]} rows, x has {n}")
        if self.x.ndim != 2 or self.theta.ndim != 2:
            raise ValueError("x and theta must be 2-D (n_rows, n_features)")
        if self.x_names and len(self.x_names) != self.x.shape[1]:
            raise ValueError(
                f"x_names has {len(self.x_names)} entries, x has {self.x.shape[1]} columns")
        if self.theta_names and len(self.theta_names) != self.theta.shape[1]:
            raise ValueError(
                f"theta_names has {len(self.theta_names)} entries, theta has "
                f"{self.theta.shape[1]} columns")
        bad = set(np.unique(self.split)) - set(SPLITS)
        if bad:
            raise ValueError(f"unknown split labels {sorted(bad)}; expected {SPLITS}")
        if np.any(~np.isfinite(self.weight)) or np.any(np.asarray(self.weight) < 0):
            raise ValueError("weight must be finite and non-negative")

    # -- shape ---------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def x_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def theta_dim(self) -> int:
        return int(self.theta.shape[1])

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.group_id).shape[0])

    @property
    def weights_are_unit(self) -> bool:
        """Exact, not approximate: a weighting scheme is either live or it is not."""
        return bool(np.all(np.asarray(self.weight) == 1.0))

    @property
    def is_single_source(self) -> bool:
        return int(np.unique(self.source_id).shape[0]) <= 1

    # -- construction --------------------------------------------------------
    @classmethod
    def unweighted(cls, x, theta, group_id, split, *, x_names=(), theta_names=(),
                   log_q=None, source_id=0) -> "RatioDataset":
        """Single-source, unweighted dataset -- the columns filled EXPLICITLY.

        ``log_q`` defaults to zeros, which means "an improper/unspecified proposal". That
        is honest for a single pool (importance weights to any target are then defined
        only up to a constant, which cancels) but it is NOT a licence to combine two pools
        built this way -- their proposals differ and the constants do not cancel.
        :func:`concatenate` refuses that combination.
        """
        n = np.asarray(x).shape[0]
        return cls(
            x=np.asarray(x, np.float32), theta=np.asarray(theta, np.float32),
            group_id=np.asarray(group_id, np.int64),
            weight=np.ones(n, np.float32),
            log_q=(np.zeros(n, np.float64) if log_q is None
                   else np.asarray(log_q, np.float64)),
            source_id=np.full(n, int(source_id), np.int32),
            split=np.asarray(split), x_names=tuple(x_names),
            theta_names=tuple(theta_names),
        )

    # -- views ---------------------------------------------------------------
    def select(self, mask_or_rows) -> "RatioDataset":
        # a slice is passed through untouched so it stays a VIEW (see split_view); anything
        # else is coerced to an index/mask array, which necessarily copies
        sel = mask_or_rows if isinstance(mask_or_rows, slice) else np.asarray(mask_or_rows)
        return replace(
            self, x=self.x[sel], theta=self.theta[sel], group_id=self.group_id[sel],
            weight=self.weight[sel], log_q=self.log_q[sel],
            source_id=self.source_id[sel], split=self.split[sel],
        )

    def split_view(self, name: str) -> "RatioDataset":
        """Rows of one split.

        Fast path: when a split occupies a CONTIGUOUS run of rows -- which it does whenever
        the dataset was written split-ordered, the natural layout for anything built from
        per-split files -- this slices instead of boolean-indexing. On a memory-mapped
        dataset that is the difference between a free view and a multi-gigabyte host copy
        (at 10^8 rows, mask-indexing every column materializes the whole split in RAM only
        to hand it straight to the device). Falls back to the mask for interleaved splits,
        which is what a hash-assigned split produces.
        """
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        mask = self.split == name
        idx = np.flatnonzero(mask)
        if idx.size and idx.size == idx[-1] - idx[0] + 1:
            return self.select(slice(int(idx[0]), int(idx[-1]) + 1))
        return self.select(mask)

    def split_sizes(self) -> dict:
        return {s: int(np.count_nonzero(self.split == s)) for s in SPLITS}

    def describe(self) -> dict:
        """Small JSON-able summary; goes into the training receipt."""
        return {
            "n_rows": self.n_rows, "n_groups": self.n_groups,
            "x_dim": self.x_dim, "theta_dim": self.theta_dim,
            "x_names": list(self.x_names), "theta_names": list(self.theta_names),
            "splits": self.split_sizes(),
            "weights_are_unit": self.weights_are_unit,
            "n_sources": int(np.unique(self.source_id).shape[0]),
            "source_ids": [int(s) for s in np.unique(self.source_id)],
        }


def hash_split(group_key, *, train_frac: float = 0.7, val_frac: float = 0.15,
               salt: str = "hitman-ratio-v1") -> np.ndarray:
    """GROUP-level train/val/test split from a stable hash of each group's key.

    Hashing a durable per-group key -- not drawing from a Generator -- is what makes the
    split reproducible across reruns, resamplings, subsamples and read orders. Two runs
    that see the same group put it on the same side even if they read the pool in a
    different order or select a different subset.

    ``group_key`` is one key per ROW; every row of a group must carry the same key (its
    shower seed, say) so the whole group lands on one side. That is the property a
    row-level split cannot have, and its absence is a silent leak: with many rows per
    group, val rows are near-duplicates of train rows and the val loss reads far too good.
    """
    import hashlib

    keys = np.asarray(group_key)
    out = np.empty(keys.shape[0], dtype="<U5")
    cache: dict = {}
    for i, k in enumerate(keys):
        lab = cache.get(int(k))
        if lab is None:
            h = hashlib.sha256(f"{salt}:{int(k)}".encode()).hexdigest()
            u = int(h[:16], 16) / float(16 ** 16)
            lab = "train" if u < train_frac else ("val" if u < train_frac + val_frac
                                                  else "test")
            cache[int(k)] = lab
        out[i] = lab
    return out


def concatenate(datasets, *, require_proposal: bool = True) -> RatioDataset:
    """Combine datasets from different sources into one multi-source dataset.

    ``source_id`` is renumbered contiguously so downstream per-source reporting is dense.
    ``group_id`` is offset per source so groups from different pools can never collide --
    a collision would let a negative be drawn from "the same" group across pools, which is
    the exact bug the group-aware pairing exists to prevent.

    ``require_proposal`` refuses to combine datasets whose ``log_q`` is identically zero,
    i.e. that never recorded a proposal density. Combining pools drawn from DIFFERENT
    proposals without knowing what they were is not a reweighting problem you can fix
    later -- the information is simply gone. Pass ``require_proposal=False`` only when you
    know the sources share one proposal exactly.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("concatenate: no datasets given")
    first = datasets[0]
    for d in datasets[1:]:
        if d.x_dim != first.x_dim or d.theta_dim != first.theta_dim:
            raise ValueError(
                f"feature dimensions differ: ({first.x_dim}, {first.theta_dim}) vs "
                f"({d.x_dim}, {d.theta_dim})")
        if d.x_names != first.x_names or d.theta_names != first.theta_names:
            raise ValueError(
                f"feature NAMES differ between sources: {first.x_names}/{first.theta_names} "
                f"vs {d.x_names}/{d.theta_names}. Same width is not the same meaning.")
    if require_proposal and len(datasets) > 1:
        flat = [i for i, d in enumerate(datasets) if np.all(d.log_q == 0.0)]
        if flat:
            raise ValueError(
                f"datasets {flat} have log_q identically zero -- no proposal density was "
                f"recorded, so they cannot be reweighted onto a common target. Record "
                f"log_q at planning time (the pool's log_q_proposal column), or pass "
                f"require_proposal=False if you KNOW all sources share one proposal.")
    offs, gid_off = [], 0
    for s, d in enumerate(datasets):
        offs.append(np.asarray(d.group_id) + gid_off)
        gid_off += int(np.asarray(d.group_id).max()) + 1 if d.n_rows else 0
        del s
    return RatioDataset(
        x=np.concatenate([d.x for d in datasets]),
        theta=np.concatenate([d.theta for d in datasets]),
        group_id=np.concatenate(offs),
        weight=np.concatenate([d.weight for d in datasets]),
        log_q=np.concatenate([d.log_q for d in datasets]),
        source_id=np.concatenate([np.full(d.n_rows, s, np.int32)
                                  for s, d in enumerate(datasets)]),
        split=np.concatenate([d.split for d in datasets]),
        x_names=first.x_names, theta_names=first.theta_names,
    )


def effective_sample_size(weight, source_id=None) -> dict:
    """Kish ESS/N overall and per source.

    ESS/N = (sum w)^2 / (n * sum w^2). It is 1 for equal weights and collapses toward 0 as
    a few rows dominate. Reported per source because that is where reweighting variance
    shows up first: one badly-mismatched pool can carry a healthy pooled number.
    """
    w = np.asarray(weight, dtype=np.float64)

    def _ess(v):
        if v.size == 0 or not np.any(v > 0):
            return 0.0
        return float(v.sum() ** 2 / (v.size * np.sum(v ** 2)))

    out = {"overall": _ess(w)}
    if source_id is not None:
        sid = np.asarray(source_id)
        out["per_source"] = {int(s): _ess(w[sid == s]) for s in np.unique(sid)}
    return out


def importance_weights(log_p_target, log_q_proposal, *, sim_weight=None,
                       w_max: Optional[float] = 100.0) -> np.ndarray:
    """Mean-1 importance weights retargeting a proposal to a target prior.

    ``w = exp(log_p_target - log_q_proposal) * sim_weight``, normalized to mean 1 and
    bounded by ``w_max``. Clipping is a variance guard, not cosmetics: one row with a huge
    weight is a single-sample estimator wearing a dataset's clothes.

    The two conditions -- ``mean(w) == 1`` and ``max(w) <= w_max`` -- fight each other: a
    single clip lowers the mean, and renormalizing afterwards scales the clipped values
    back ABOVE the bound. So the clip/renormalize pair is iterated to a fixed point
    (monotone, converges in a handful of passes) rather than applied once, which is the
    subtle bug in the naive version -- it reports a bound it does not actually hold to.
    ``w_max`` is therefore a genuine bound on the weight relative to the mean.

    NOTE this returns the weight only. A theta-dependent weight also has to move the
    MARGINAL DRAW -- weighting only the joint term retargets the numerator and leaves the
    denominator alone, which is not the ratio anyone wants. See
    :func:`hitman.ratio.train.fit_ratio`'s ``pairing``/``weight`` handling.
    """
    logw = np.asarray(log_p_target, np.float64) - np.asarray(log_q_proposal, np.float64)
    logw -= logw.max()                      # overflow guard; the constant cancels below
    w = np.exp(logw)
    if sim_weight is not None:
        w = w * np.asarray(sim_weight, np.float64)
    w = w / max(w.mean(), 1e-300)
    if w_max is not None:
        cap = float(w_max)
        for _ in range(100):
            over = w > cap
            if not over.any():
                break
            w = np.minimum(w, cap)
            w = w / max(w.mean(), 1e-300)
        else:  # pragma: no cover - only if a pathological weight set will not converge
            w = np.minimum(w, cap)
    return w.astype(np.float32)
