"""L2 receipt battery: forward-model + inference receipts for a trained model dir.

Run as ``python -m hitman.receipts <model_dir> <testpoint_spec>...`` to produce a
machine-readable ``receipts.json``; compare against a baseline with
:mod:`hitman.receipts.thresholds`. The numerical primitives (reweighting, chi2, score
identity, MLE reduction) are importable and unit-tested independently of any model.
"""

from hitman.receipts.chi2 import Chi2Result, chi2_comparison
from hitman.receipts.reweight import (
    PoolWeights,
    group_fractions,
    reweighted_density,
    self_normalized_weights,
    weighted_fraction,
)
from hitman.receipts.score import ScoreIdentity, score_identity
from hitman.receipts.mle import MLEResult, mle_receipt

__all__ = [
    "PoolWeights",
    "self_normalized_weights",
    "weighted_fraction",
    "reweighted_density",
    "group_fractions",
    "Chi2Result",
    "chi2_comparison",
    "ScoreIdentity",
    "score_identity",
    "MLEResult",
    "mle_receipt",
]
