"""Neural posterior estimation (NPE) arm: DeepSets encoder + conditional spline flow.

An amortized q_psi(theta | x) trained by forward-KL MLE, cross-checking the NRE
likelihood-ratio surrogate. The encoder summarizes an event's hits into a
hypothesis-free context; the flow is an exact-density conditional normalizing flow over
the 7-D hypothesis with periodic azimuth handling.
"""

from hitman._compat import deprecated_getattr
from hitman.npe.encoder import DeepSetsEncoder
from hitman.npe.flow import ConditionalFlow
from hitman.npe.loss import encode, flow_nll, npe_loss
from hitman.npe.sbc import sbc_ranks, tarp_coverage

# The batching and the training loop are water-Cherenkov: they gather through a PMT
# geometry table, shift theta[5] for the coherent time augmentation, and read a HitStore's
# offsets. They moved to hitman.wc.npe; served lazily so importing this lane does not drag
# in the application.
__getattr__ = deprecated_getattr(__name__, {
    "build_npe_spec": "hitman.wc.npe.batch",
    "make_npe_batch": "hitman.wc.npe.batch",
    "make_fixed_batch": "hitman.wc.npe.batch",
    "fit_npe": "hitman.wc.npe.train",
    "NpeFitResult": "hitman.wc.npe.train",
})

__all__ = [
    "DeepSetsEncoder", "ConditionalFlow", "encode", "flow_nll", "npe_loss",
    "build_npe_spec", "make_npe_batch", "make_fixed_batch", "fit_npe", "NpeFitResult",
    "sbc_ranks", "tarp_coverage",
]
