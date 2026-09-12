"""Detector-agnostic receipt core.

Reweighting (self-normalized importance weights), chi2 comparison, the score identity,
the receipt JSON schema and the threshold policy engine are all detector-agnostic and
synthetic-net tested. The water-Cherenkov forward observables (per-PMT charge, z-ring
time), the model-dir runner and the (model, data) harness moved to
:mod:`hitman.wc.receipts`; ``percentile_strata`` -- the generic equal-occupancy strata
primitive underneath ``z_rings`` -- stayed here as :mod:`hitman.receipts.strata`.
"""

from hitman._compat import deprecated_getattr
from hitman.receipts.chi2 import Chi2Result, chi2_comparison
from hitman.receipts.reweight import (
    PoolWeights, group_fractions, reweighted_density, self_normalized_weights,
    weighted_fraction,
)
from hitman.receipts.score import ScoreIdentity, score_identity
from hitman.receipts.strata import percentile_strata

__getattr__ = deprecated_getattr(__name__, {
    "MLEResult": "hitman.wc.receipts.mle",
    "mle_receipt": "hitman.wc.receipts.mle",
})

__all__ = [
    "PoolWeights", "self_normalized_weights", "weighted_fraction", "reweighted_density",
    "group_fractions", "Chi2Result", "chi2_comparison", "ScoreIdentity", "score_identity",
    "percentile_strata", "MLEResult", "mle_receipt",
]
