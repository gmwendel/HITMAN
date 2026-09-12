"""The neural-ratio lane: grouped, weighted, multi-source ratio estimation.

``log r(x | theta) = log p(x | theta) - log p(x)`` learned as a binary classifier's
logit. What distinguishes this lane from a generic classifier trainer is that it takes
the structure of simulation data seriously:

* rows are GROUPED (many observations share one hypothesis), so splits and marginal
  negatives must be group-aware -- :mod:`hitman.ratio.pairing`;
* rows are WEIGHTED (prescale, thinning) and come from SOURCES with different proposal
  densities, so reweighting is a column operation -- :mod:`hitman.ratio.dataset`;
* a trained model is only an estimator inside its training support, and that support has
  to travel with the weights -- :mod:`hitman.ratio.support`,
  :mod:`hitman.ratio.bundle`.

Typical use::

    ds = RatioDataset.unweighted(x, theta, group_id, split, x_names=..., theta_names=...)
    model = build_ratio_model(ds, key=k)
    result = fit_ratio(model, ds, key=k, pairing="group_shift")
    support = compute_support(result.model, ds)
    save_bundle("run/bundle", RatioBundle(model=result.model, support=support))
"""

from hitman.ratio.bundle import RatioBundle, load_bundle, save_bundle
from hitman.ratio.dataset import (
    RatioDataset, concatenate, effective_sample_size, hash_split, importance_weights,
)
from hitman.ratio.model import RatioModel, Standardizer, build_ratio_model, n_parameters
from hitman.ratio.pairing import (
    GroupDerangement, GroupIndex, GroupShift, NegativePairing, RowPermutation,
    build_group_index, get_pairing,
)
from hitman.ratio.support import FeatureSupport, compute_support, make_barrier, trust_radius
from hitman.ratio.train import FitResult, fit_ratio, ratio_bce

__all__ = [
    "RatioDataset", "hash_split", "concatenate", "effective_sample_size",
    "importance_weights",
    "RatioModel", "Standardizer", "build_ratio_model", "n_parameters",
    "NegativePairing", "GroupIndex", "build_group_index", "get_pairing",
    "RowPermutation", "GroupShift", "GroupDerangement",
    "FeatureSupport", "compute_support", "trust_radius", "make_barrier",
    "fit_ratio", "FitResult", "ratio_bce",
    "RatioBundle", "save_bundle", "load_bundle",
]
