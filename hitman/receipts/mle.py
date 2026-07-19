"""MLE bias / resolution / pull receipt on a fixed-hypothesis test ensemble.

Library form of the ``mle_survey2`` / ``validate_point`` summaries. Pure numpy: the
caller runs the (jitted, vmapped) multistart MLE and passes in the per-event fits and
Fisher sigmas; this reduces them to per-parameter bias, resolution (IQR/1.35, robust to
the direction tail), and pull width (``std((fit - truth) / sigma)`` — ~1 iff the
surrogate's local curvature matches its actual error, "curvature honesty").
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np

_DEFAULT_NAMES = ("x", "y", "z", "zenith", "azimuth", "t", "E")


class ParamStat(NamedTuple):
    name: str
    bias: float          # median(fit - truth)
    resolution: float    # IQR/1.35 of (fit - truth)
    pull_sigma: float    # std of clipped (fit - truth)/sigma (nan if sigmas absent)


class MLEResult(NamedTuple):
    params: Sequence[ParamStat]
    psi_median_deg: float   # median opening angle between fit and true direction
    n_events: int
    mean_nhit: float


def _iqr_std(v):
    return float(np.subtract(*np.percentile(v, [75, 25])) / 1.35)


def opening_angles_deg(fits, truth):
    """Opening angle (deg) between each fit direction and the truth direction."""
    fits = np.asarray(fits)
    truth = np.asarray(truth)
    d_true = np.array(
        [
            np.sin(truth[3]) * np.cos(truth[4]),
            np.sin(truth[3]) * np.sin(truth[4]),
            np.cos(truth[3]),
        ]
    )
    d_fit = np.stack(
        [
            np.sin(fits[:, 3]) * np.cos(fits[:, 4]),
            np.sin(fits[:, 3]) * np.sin(fits[:, 4]),
            np.cos(fits[:, 3]),
        ],
        axis=1,
    )
    return np.degrees(np.arccos(np.clip(d_fit @ d_true, -1.0, 1.0)))


def mle_receipt(
    fits,
    truth,
    sigmas: Optional[np.ndarray] = None,
    mean_nhit: float = float("nan"),
    names: Sequence[str] = _DEFAULT_NAMES,
    pull_clip: float = 8.0,
) -> MLEResult:
    """Reduce per-event MLE fits to bias/resolution/pull per parameter.

    Parameters
    ----------
    fits : (n_events, 7)
        Best-fit hypotheses (x, y, z, zenith, azimuth, t, E).
    truth : (7,)
        The generating hypothesis.
    sigmas : (n_events, 7), optional
        Per-event Fisher sigmas; if given, pull widths are reported.
    mean_nhit : float
        Mean hit multiplicity of the ensemble (carried through for context).
    """
    fits = np.asarray(fits, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    sigmas = None if sigmas is None else np.asarray(sigmas, dtype=np.float64)

    stats = []
    for i, name in enumerate(names):
        resid = fits[:, i] - truth[i]
        if sigmas is not None and name not in ("zenith", "azimuth"):
            pull = resid / np.maximum(sigmas[:, i], 1e-9)
            pull = pull[np.abs(pull) < pull_clip]
            pull_sigma = float(pull.std()) if len(pull) else float("nan")
        else:
            pull_sigma = float("nan")
        stats.append(
            ParamStat(
                name=name,
                bias=float(np.median(resid)),
                resolution=_iqr_std(resid),
                pull_sigma=pull_sigma,
            )
        )
    psi = opening_angles_deg(fits, truth)
    return MLEResult(
        params=stats,
        psi_median_deg=float(np.median(psi)),
        n_events=len(fits),
        mean_nhit=float(mean_nhit),
    )
