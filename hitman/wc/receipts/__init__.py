"""Water-Cherenkov forward receipts: per-PMT charge, z-ring time, the model-dir runner.

The detector-agnostic numeric core (chi2, self-normalized reweighting, the score
identity, the receipt JSON schema, the threshold policy) stayed in ``hitman.receipts``.
What lives here needs a sensor grid or a HitStore.
"""

from hitman.wc.receipts.forward import (
    NTotResult, RingTimeResult, implied_ntot_pmf, ntot_receipt,
    per_pmt_charge_receipt, per_ring_time_receipt, toa_receipt,
)
from hitman.wc.receipts.harness import (
    NREReceiptModel, ReceiptModel, compute_forward_block, forward_block_from_model,
    percentile_strata, z_rings,
)
from hitman.wc.receipts.mle import MLEResult, mle_receipt, opening_angles_deg
from hitman.wc.receipts.runner import build_marginal_pool, git_sha, load_models, run_model_dir

__all__ = [
    "NTotResult", "RingTimeResult", "implied_ntot_pmf", "ntot_receipt",
    "per_pmt_charge_receipt", "per_ring_time_receipt", "toa_receipt",
    "NREReceiptModel", "ReceiptModel", "compute_forward_block",
    "forward_block_from_model", "percentile_strata", "z_rings",
    "MLEResult", "mle_receipt", "opening_angles_deg",
    "build_marginal_pool", "git_sha", "load_models", "run_model_dir",
]
