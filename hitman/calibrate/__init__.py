"""Estimator calibration: Godambe/sandwich adjustment (detector-agnostic).

The per-PMT/time marginal-grid Z(theta) normalization is water-Cherenkov and moved to
:mod:`hitman.wc.calibrate.normalization`.
"""

from hitman._compat import deprecated_getattr
from hitman.calibrate.godambe import (
    SandwichResult, adjust_samples, adjustment_matrix, estimate_sandwich,
)

__getattr__ = deprecated_getattr(__name__, {
    n: "hitman.wc.calibrate.normalization" for n in
    ("MarginalGrid", "ChargeGrid", "build_marginal_grid", "build_charge_grid",
     "make_z_fn", "make_z_charge_fn", "z_of_theta", "znll")
})

__all__ = ["estimate_sandwich", "adjustment_matrix", "adjust_samples", "SandwichResult"]
