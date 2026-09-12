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
    # Stable sigmoid: 1/(1+exp(-x)) overflows for large NEGATIVE x (exp of a large
    # positive). Branching on the sign keeps the exponent negative in both cases. Values
    # are identical to the naive form wherever the naive form does not overflow, and equal
    # to its correct limit where it does -- so this is a warning fix, not a numerics change.
    prob = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.abs(logits))),
        np.exp(-np.abs(logits)) / (1.0 + np.exp(-np.abs(logits))),
    )
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


def expected_calibration_error(joint_logits, marginal_logits, n_bins: int = 15) -> float:
    """Just the ECE scalar from :func:`reliability_curve` -- the training-loop metric."""
    return reliability_curve(joint_logits, marginal_logits, n_bins)[2]


def roc_auc(joint_logits, marginal_logits) -> float:
    """AUC separating joint (label 1) from marginal (label 0) pairs.

    Rank-sum (Mann-Whitney U) identity, so it is exact rather than a threshold sweep, and
    ties get average ranks. Implemented on numpy rather than via scipy so the library's
    training loop has no scipy dependency.
    """
    pos = np.asarray(joint_logits, dtype=np.float64).ravel()
    neg = np.asarray(marginal_logits, dtype=np.float64).ravel()
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = np.argsort(both, kind="stable")
    ranks = np.empty(both.size, dtype=np.float64)
    ranks[order] = np.arange(1, both.size + 1, dtype=np.float64)
    # average ranks within tie groups (the U identity assumes midranks)
    s = both[order]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum_pos = float(ranks[:n_pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def self_normalization(marginal_logits):
    """Mean estimated ratio over marginal pairs; exactly 1 for the optimal classifier.

    E_{p(x)p(theta)}[r(x, theta)] = 1. Deviations flag a mis-normalized (typically
    overconfident) ratio estimator.
    """
    logits = np.asarray(marginal_logits, dtype=np.float64)
    return float(np.mean(np.exp(logits)))


def fit_temperature(joint_logits, marginal_logits, n_grid: int = 400):
    """Fit a scalar temperature ``tau`` that recalibrates a ratio estimator.

    The recalibrated logit is ``logit / tau``; ``tau`` minimizes the composed
    classification BCE over joint (label 1) and marginal (label 0) logits::

        L(beta) = mean softplus(-beta * l_joint) + mean softplus(beta * l_marginal),
        beta = 1 / tau .

    ``L`` is convex in ``beta`` (a sum of softplus-of-linear terms), so a log-spaced
    grid scan followed by parabolic refinement finds the global optimum without SciPy.
    An estimator that is already calibrated returns ``tau ~ 1``; an overconfident one
    returns ``tau > 1``. This is the calibration knob of TODO item A.1 and the
    injectable miscalibration probe of the L1 temperature closure test.
    """
    lj = np.asarray(joint_logits, dtype=np.float64)
    lm = np.asarray(marginal_logits, dtype=np.float64)

    def loss(beta):
        return np.mean(np.logaddexp(0.0, -beta * lj)) + np.mean(np.logaddexp(0.0, beta * lm))

    betas = np.exp(np.linspace(np.log(1e-2), np.log(1e2), n_grid))
    losses = np.array([loss(b) for b in betas])
    i = int(np.argmin(losses))
    lo = max(i - 1, 0)
    hi = min(i + 1, n_grid - 1)
    # parabolic interpolation in log-beta for a sub-grid optimum
    xl, xm, xh = np.log(betas[lo]), np.log(betas[i]), np.log(betas[hi])
    yl, ym, yh = losses[lo], losses[i], losses[hi]
    denom = (xl - xm) * (xl - xh) * (xm - xh)
    if denom != 0 and lo != hi:
        a = (xh * (ym - yl) + xm * (yl - yh) + xl * (yh - ym)) / denom
        b = (xh**2 * (yl - ym) + xm**2 * (yh - yl) + xl**2 * (ym - yh)) / denom
        log_beta = -b / (2 * a) if a > 0 else xm
    else:
        log_beta = xm
    return float(1.0 / np.exp(log_beta))


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
