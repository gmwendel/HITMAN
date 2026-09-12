"""Godambe/sandwich curvature correction for the composite per-hit likelihood.

The event log-likelihood sums per-hit log-ratios as if hits were independent; they
are not (shared vertex/track fluctuations — the same correlations behind Fano > 1
counts). Composite-likelihood theory (Varin-Reid-Firth 2011) then says the MLE
scatter is the inverse Godambe information G = H J^-1 H (H = sensitivity = mean
negative Hessian of the per-event log-likelihood; J = variability = covariance of
per-event scores), while naive Fisher errors report H^-1 — overconfident whenever
J != H. Measured on this detector the effect reproduces the pull pattern exactly
(design report, Addendum 9): E honest (single charge term), direction worst
(coherent ring movement).

``adjust_samples`` is the open-faced-sandwich style post-hoc fix (Shaby,
arXiv:1204.3687): a linear map around the mode that transports sample covariance
from H^-1 to the sandwich H^-1 J H^-1 — drop-in for NUTS output. Application to a
neural per-hit surrogate is, per the 2026-07-18 literature review, unpublished.
"""

from typing import NamedTuple

import numpy as np


class SandwichResult(NamedTuple):
    H: np.ndarray          # sensitivity (mean negative Hessian), (d, d)
    J: np.ndarray          # variability (score covariance), (d, d)
    fisher_cov: np.ndarray  # H^-1 — what naive Fisher reports
    godambe_cov: np.ndarray  # H^-1 J H^-1 — the honest asymptotic covariance
    pull_prediction: np.ndarray  # per-parameter sqrt(godambe_ii / fisher_ii)


def estimate_sandwich(scores: np.ndarray, hessians: np.ndarray,
                      rcond: float = 1e-6) -> SandwichResult:
    """Estimate H, J and the sandwich from per-event scores/Hessians at a fixed theta.

    Parameters
    ----------
    scores : (n_events, d) — per-event gradient of the event LOG-likelihood.
    hessians : (m_events, d, d) — per-event Hessian of the event log-likelihood
        (a subsample is fine; only the mean enters).
    rcond : eigendirections of H below ``rcond * max_eig`` are treated as
        UNIDENTIFIED (e.g. zenith at a coordinate pole): the sandwich is formed in
        the identified subspace only, and ``pull_prediction`` is NaN along near-null
        axes. Without this cut a tiny-but-nonzero pole eigenvalue inflates the whole
        adjustment matrix (observed: zen180 receipt contaminating all six parameters).
    """
    J = np.atleast_2d(np.cov(np.asarray(scores).T))
    H = -np.mean(np.asarray(hessians), axis=0)
    H = 0.5 * (H + H.T)
    w, v = np.linalg.eigh(H)
    keep = w > rcond * np.max(w)
    winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    fisher = (v * winv) @ v.T
    godambe = fisher @ J @ fisher
    f_diag, g_diag = np.diag(fisher), np.diag(godambe)
    identified = f_diag > rcond * np.max(f_diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(identified,
                        np.sqrt(np.clip(g_diag, 0, None) / np.where(identified, f_diag, 1.0)),
                        np.nan)
    return SandwichResult(H=H, J=J, fisher_cov=fisher, godambe_cov=godambe,
                          pull_prediction=pull)


def _msqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def adjustment_matrix(sw: SandwichResult) -> np.ndarray:
    """Linear map A with A C_fisher A^T = C_godambe, chosen as the OPTIMAL-TRANSPORT
    (minimal-displacement) pairing A = Cf^-1/2 (Cf^1/2 Cg Cf^1/2)^1/2 Cf^-1/2.

    Any A with A Cf A^T = Cg is valid up to an orthogonal factor, and the choice
    matters in practice: real chains have covariance != Cf (per-event curvature,
    non-Gaussianity), so a pairing with a rotation component leaks one parameter's
    correction into another (observed: E coverage overshooting at zen180 under the
    naive square-root pairing). The OT map is symmetric positive-definite — zero
    rotation — and reduces to the intuitive per-axis stretch when the covariances
    commute. On the unidentified null space of H (zeroed in both covariances by
    ``estimate_sandwich``) the map acts as the IDENTITY.
    """
    w, v = np.linalg.eigh(0.5 * (sw.fisher_cov + sw.fisher_cov.T))
    keep = w > 1e-12 * np.max(np.abs(w))
    P = v[:, keep]
    cf_r = P.T @ sw.fisher_cov @ P
    cg_r = P.T @ sw.godambe_cov @ P
    s_f = _msqrt(cf_r)
    s_f_inv = np.linalg.inv(s_f)
    A_r = s_f_inv @ _msqrt(s_f @ cg_r @ s_f) @ s_f_inv
    return P @ A_r @ P.T + v[:, ~keep] @ v[:, ~keep].T


def adjust_samples(samples: np.ndarray, mode: np.ndarray, sw: SandwichResult) -> np.ndarray:
    """Open-faced-sandwich style post-hoc adjustment of posterior samples.

    Recenters at the mode and stretches by ``adjustment_matrix`` so the sample
    covariance attains the Godambe (honest) covariance. Location and skewness
    structure are preserved up to the linear map; only curvature is corrected.
    """
    A = adjustment_matrix(sw)
    centered = np.asarray(samples) - np.asarray(mode)
    return np.asarray(mode) + centered @ A.T
