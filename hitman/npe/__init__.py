"""Neural posterior estimation (NPE) arm: DeepSets encoder + conditional spline flow.

An amortized q_psi(theta | x) trained by forward-KL MLE, cross-checking the NRE
likelihood-ratio surrogate. The encoder summarizes an event's hits into a
hypothesis-free context; the flow is an exact-density conditional normalizing flow over
the 7-D hypothesis with periodic azimuth handling.
"""

from hitman.npe.encoder import DeepSetsEncoder
from hitman.npe.flow import ConditionalFlow
from hitman.npe.loss import encode, flow_nll, npe_loss
from hitman.npe.batch import build_npe_spec, make_npe_batch, make_fixed_batch
from hitman.npe.train import fit_npe, NpeFitResult
from hitman.npe.sbc import sbc_ranks, tarp_coverage

__all__ = [
    "DeepSetsEncoder", "ConditionalFlow", "encode", "flow_nll", "npe_loss",
    "build_npe_spec", "make_npe_batch", "make_fixed_batch", "fit_npe", "NpeFitResult",
    "sbc_ranks", "tarp_coverage",
]
