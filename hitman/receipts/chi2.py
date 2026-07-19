"""chi2 comparison of a surrogate prediction against an MC reference.

Every forward receipt reduces to the same question: does the surrogate curve agree
with the 50k-event MC within the *combined* error bar (Poisson MC + importance-sampling
surrogate)? This module is that comparison, factored out of the plotting scripts so it
can be unit-tested and thresholded.
"""

from typing import NamedTuple, Optional

import numpy as np


class Chi2Result(NamedTuple):
    """Outcome of a binned surrogate-vs-MC comparison.

    Attributes
    ----------
    chi2 : float
    dof : int
        Number of compared bins minus ``ddof``.
    chi2_dof : float
        ``chi2 / dof`` — the headline number (want ~1).
    mc_var_share : float
        Median fraction of the combined variance contributed by the MC (Poisson) side.
        Near 1 => the test is MC-statistics-limited (surrogate error negligible);
        near 0 => the surrogate's own IS error dominates and the test is weak.
    pulls : (n_selected,) float64
        ``(pred - obs) / sqrt(var)`` for the compared bins.
    n_selected : int
    """

    chi2: float
    dof: int
    chi2_dof: float
    mc_var_share: float
    pulls: np.ndarray
    n_selected: int


def chi2_comparison(
    obs,
    obs_err,
    pred,
    pred_err,
    mask: Optional[np.ndarray] = None,
    ddof: int = 1,
    floor: float = 1e-12,
) -> Chi2Result:
    """Combined-error chi2 of ``pred`` against reference ``obs``.

    Parameters
    ----------
    obs, obs_err : (n,)
        MC reference values and their (Poisson) errors.
    pred, pred_err : (n,)
        Surrogate prediction and its (importance-sampling) errors.
    mask : (n,) bool, optional
        Bins to include (e.g. MC occupancy > 5). Defaults to all.
    ddof : int
        Degrees of freedom subtracted (1 for an overall-normalization constraint).
    floor : float
        Lower clip on the combined variance to avoid division by zero on empty bins.
    """
    obs = np.asarray(obs, dtype=np.float64)
    obs_err = np.asarray(obs_err, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    pred_err = np.asarray(pred_err, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(obs), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    o, oe, p, pe = obs[mask], obs_err[mask], pred[mask], pred_err[mask]
    var = np.maximum(oe**2 + pe**2, floor)
    chi2 = float(np.sum((o - p) ** 2 / var))
    n_sel = int(mask.sum())
    dof = max(n_sel - ddof, 1)
    mc_var_share = float(np.median(oe**2 / var)) if n_sel else float("nan")
    pulls = (p - o) / np.sqrt(var)
    return Chi2Result(
        chi2=chi2,
        dof=dof,
        chi2_dof=chi2 / dof,
        mc_var_share=mc_var_share,
        pulls=pulls,
        n_selected=n_sel,
    )
