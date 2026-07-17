"""Consistency and calibration diagnostics for the learned ratio estimator.

Post-2022 SBI hygiene: a surrogate likelihood should ship with receipts —
classifier reliability, self-normalization of the ratio, and simulation-based
calibration (SBC) rank/coverage checks on held-out simulations
(cf. Hermans et al. arXiv:2110.06581; Talts et al. arXiv:1804.06788).

These run offline on numpy arrays.
"""

import numpy as np


def reliability_curve(joint_logits, marginal_logits, n_bins: int = 15):
    """Classifier calibration curve and expected calibration error (ECE).

    Returns
    -------
    bin_confidence : (n_bins,) mean predicted P(joint) per bin (nan for empty bins)
    bin_accuracy : (n_bins,) empirical fraction of joint pairs per bin
    ece : float, |confidence − accuracy| weighted by bin occupancy
    """
    logits = np.concatenate([np.asarray(joint_logits), np.asarray(marginal_logits)])
    labels = np.concatenate(
        [np.ones(len(joint_logits)), np.zeros(len(marginal_logits))]
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    ece = 0.0
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            bin_confidence[b] = prob[mask].mean()
            bin_accuracy[b] = labels[mask].mean()
            ece += mask.mean() * abs(bin_confidence[b] - bin_accuracy[b])
    return bin_confidence, bin_accuracy, float(ece)


def self_normalization(marginal_logits):
    """Mean estimated ratio over marginal pairs; exactly 1 for the optimal classifier.

    E_{p(x)p(theta)}[r(x, theta)] = 1. Deviations flag a mis-normalized (typically
    overconfident) ratio estimator.
    """
    logits = np.asarray(marginal_logits, dtype=np.float64)
    return float(np.mean(np.exp(logits)))


def sbc_ranks(posterior_samples, truths):
    """Simulation-based calibration ranks.

    Parameters
    ----------
    posterior_samples : (n_sims, n_draws, dim)
        Posterior draws (e.g. NUTS) for each held-out simulated event.
    truths : (n_sims, dim)
        The parameters that generated each event.

    Returns
    -------
    (n_sims, dim) integer ranks in [0, n_draws]; uniform iff the posterior is calibrated.
    """
    posterior_samples = np.asarray(posterior_samples)
    truths = np.asarray(truths)
    return np.sum(posterior_samples < truths[:, None, :], axis=1)


def expected_coverage(ranks, n_draws: int, levels=None):
    """Empirical coverage of central credible intervals vs nominal credibility.

    A calibrated posterior lies on the diagonal; above = conservative,
    below = overconfident.

    Returns
    -------
    levels : (n_levels,) nominal credibility
    coverage : (n_levels, dim) empirical coverage per parameter
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    levels = np.asarray(levels)
    u = ranks / n_draws  # ~U(0,1) if calibrated
    coverage = np.stack(
        [np.mean(np.abs(u - 0.5) <= level / 2.0, axis=0) for level in levels]
    )
    return levels, coverage
