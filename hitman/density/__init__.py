"""Pluggable density-factor library (DESIGN proposal 2).

The splinemle water-Cherenkov model couples three *exactly-normalized* likelihood factors —
a closed-form 1-D log-spline time density, a discrete sensor softmax, and an NB2/Poisson
count model with a monotone yield head — into one detector-specific ``eqx.Module``. This
package extracts those factors behind a small :class:`DensityFactor` protocol so a per-event
likelihood composes as ``count_factor x prod(mark_factors)``:

    log L(event | context) = count.log_prob(N | context)
                           + sum_objects [ sum_marks mark_k.log_prob(v_k | context) ]

Every factor normalizes in closed form (no Monte Carlo, no grid in the data dimension), so
any product of factors is itself a proper density. The splinemle WC model is reconstructed
from these pieces (:class:`SoftmaxMarkFactor` + :class:`LogSplineFactor` +
:class:`CountFactor`); ``tests/test_density_factors.py`` checks the reconstruction is
bit-for-bit (float64) equal to the monolithic ``SplineMLE``.

Downstream (e.g. the muon-station program) assembles NB2(N|Theta) x a chain of 1-D
normalized conditionals over ``(dt, alpha_r, alpha_t)`` from the SAME factors — a log-spline
factor for ``dt`` and RQ-spline factors for the bounded angles — with zero new normalization
code.
"""

from hitman.density.logspline import (
    LogSplineFactor,
    SUPPORT_FLOOR,
    density_u,
    ell_at,
    interval_log_integrals,
    log_expm1_over_x,
    log_prob_u,
    logZ_time,
)
from hitman.density.count import (
    CountFactor,
    affine_log_dispersion,
    init_phi_params,
    monotone_phi_values,
    nb2_log_count,
    phi_at,
    poisson_log_count,
)
from hitman.density.factors import (
    DensityFactor,
    SoftmaxMarkFactor,
    marked_poisson_loglik,
)

__all__ = [
    # protocol + composition
    "DensityFactor", "SoftmaxMarkFactor", "marked_poisson_loglik",
    # log-spline (continuous) factor + closed-form pieces
    "LogSplineFactor", "SUPPORT_FLOOR", "log_expm1_over_x", "interval_log_integrals",
    "logZ_time", "ell_at", "log_prob_u", "density_u",
    # count factor + closed-form pieces
    "CountFactor", "nb2_log_count", "poisson_log_count", "monotone_phi_values",
    "phi_at", "affine_log_dispersion", "init_phi_params",
]
