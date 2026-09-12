"""Self-normalized importance reweighting of a training-marginal pool.

The learned hit ratio ``r_hat(obs, theta)`` is calibrated against the training hit
marginal (with the +/-50 ns event-time augmentation), so the surrogate-implied
observation distribution at a fixed hypothesis is obtained by reweighting an
(augmented) marginal pool::

    p_hat(obs | theta) = r_hat(obs, theta) * p_marginal(obs) / Z .

These are the pure-numpy primitives underneath ``forward_check.py`` /
``forward_time_per_pmt.py``; nothing here needs a model, JAX, or ROOT — a caller
passes in the per-pool log-weights (``log r_hat``) it computed however it likes.
"""

from typing import NamedTuple

import numpy as np


class PoolWeights(NamedTuple):
    """Normalized importance weights for a marginal pool at one hypothesis.

    Attributes
    ----------
    weights : (n_pool,) float64
        Normalized weights ``w_i`` summing to 1 (self-normalized IS weights).
    self_norm : float
        ``E[r_hat] = mean(exp(logw))`` over the pool. Exactly 1 for the optimal
        classifier; a standing receipt for a mis-normalized (overconfident) ratio.
    ess : float
        Effective sample size ``1 / sum(w_i^2)``. Small ESS => the reweighted
        estimate rests on few pool members and its error bars are unreliable.
    """

    weights: np.ndarray
    self_norm: float
    ess: float


def self_normalized_weights(logw) -> PoolWeights:
    """Turn pool log-weights ``log r_hat(obs_i, theta)`` into normalized IS weights.

    The max-subtraction is for numerical stability only; the normalization divides
    it out. ``self_norm`` is computed from the raw (un-shifted) exponentials so it
    remains the honest ``E[r_hat]`` receipt.
    """
    logw = np.asarray(logw, dtype=np.float64)
    self_norm = float(np.mean(np.exp(logw)))
    w = np.exp(logw - logw.max())
    total = w.sum()
    weights = w / total if total > 0 else np.full_like(w, 1.0 / len(w))
    ess = float(1.0 / np.sum(weights**2))
    return PoolWeights(weights=weights, self_norm=self_norm, ess=ess)


def weighted_fraction(weights, member):
    """Weighted fraction of the pool in ``member`` and its importance-sampling sigma.

    ``member`` is a boolean mask over the pool. The estimator is
    ``p = sum_i w_i [i in member]`` and its variance uses the delta-method form for a
    self-normalized ratio estimator (matching the scripts' ``is_error``):
    ``var = sum_i (w_i (1_member,i - p))^2``.

    Returns
    -------
    p : float
    sigma : float
    """
    weights = np.asarray(weights, dtype=np.float64)
    member = np.asarray(member)
    p = float(weights[member].sum())
    var = float(np.sum((weights * (member.astype(np.float64) - p)) ** 2))
    return p, np.sqrt(max(var, 0.0))


def reweighted_density(pool_values, weights, bins, renormalize: bool = True):
    """Surrogate-implied density of a scalar pool quantity, with per-bin IS errors.

    Parameters
    ----------
    pool_values : (n_pool,)
        The scalar (e.g. hit time) whose density is estimated.
    weights : (n_pool,)
        Normalized IS weights from :func:`self_normalized_weights`.
    bins : (n_bins + 1,)
        Bin edges.
    renormalize : bool
        If True, rescale so the density integrates to 1 over the binned window
        (the pool may carry weight outside the window).

    Returns
    -------
    density : (n_bins,)
    density_err : (n_bins,)
    """
    pool_values = np.asarray(pool_values)
    weights = np.asarray(weights, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    width = np.diff(bins)
    density = np.zeros(len(width))
    density_err = np.zeros(len(width))
    for b in range(len(width)):
        member = (pool_values >= bins[b]) & (pool_values < bins[b + 1])
        p, e = weighted_fraction(weights, member)
        density[b] = p / width[b]
        density_err[b] = e / width[b]
    if renormalize:
        norm = float(np.sum(density * width))
        scale = 1.0 / norm if norm > 1e-12 else 0.0
        density = density * scale
        density_err = density_err * scale
    return density, density_err


def group_fractions(weights, group_ids, n_groups: int):
    """Weighted fraction (+ IS sigma) of the pool in each of ``n_groups`` groups.

    ``group_ids`` is an integer array over the pool (e.g. ``pmt_id`` or ring index).
    Vectorized equivalent of the scripts' per-PMT loop.

    Returns
    -------
    frac : (n_groups,)
    frac_err : (n_groups,)
    """
    weights = np.asarray(weights, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    frac = np.zeros(n_groups)
    frac_err = np.zeros(n_groups)
    for g in range(n_groups):
        frac[g], frac_err[g] = weighted_fraction(weights, group_ids == g)
    return frac, frac_err
