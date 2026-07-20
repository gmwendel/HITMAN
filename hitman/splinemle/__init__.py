"""run20: per-hit conditional density with closed-form normalization in continuous time.

A log-spline (piecewise-exponential) time density in TOF-residual coordinates with a
softmax sensor factor and a marked-Poisson count factor, trained by exact maximum
likelihood. Every factor normalizes analytically -- no Monte Carlo, no grid in the data
dimension. See ``hitman.splinemle.model`` for the design rationale.
"""

from hitman.splinemle.model import (
    C_MM_PER_NS,
    DEFAULT_KNOTS,
    N_COND_FEATURES,
    SUPPORT_FLOOR,
    SplineMLE,
    density_u,
    ell_at,
    interval_log_integrals,
    log_expm1_over_x,
    log_prob_u,
    logZ_time,
)
from hitman.splinemle.loss import batch_terms, event_terms, splinemle_loss
from hitman.splinemle.train import (
    EventSpec,
    FitResult,
    build_event_spec,
    fit_splinemle,
    make_event_batch,
    step_max_intermediate_gib,
)

__all__ = [
    "SplineMLE", "DEFAULT_KNOTS", "N_COND_FEATURES", "SUPPORT_FLOOR", "C_MM_PER_NS",
    "log_expm1_over_x", "interval_log_integrals", "logZ_time", "ell_at", "log_prob_u",
    "density_u", "event_terms", "batch_terms", "splinemle_loss",
    "EventSpec", "FitResult", "build_event_spec", "make_event_batch", "fit_splinemle",
    "step_max_intermediate_gib",
]
