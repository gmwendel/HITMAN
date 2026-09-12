"""Composite and hierarchical posteriors built from a per-row ratio estimator.

A ratio estimator is trained on SINGLE rows: the network sees ``(x_i, theta)`` and never a
group. Turning that into a statement about a group -- let alone about a parameter shared
across many groups -- requires an assumption the network was never asked to justify:

    log p(x_1..x_n | theta) - const  =  sum_i log r(x_i, theta)

which holds iff the rows are conditionally independent given ``theta``. Everything in this
module rests on that line, so it is written once, here, rather than re-derived inline at
each call site. :mod:`hitman.validate.toys` supplies the case where the assumption is true
by construction (and the case where it provably fails), so this code can be checked against
exact numbers instead of against itself.

The functions are grid-based rather than gradient-based on purpose. A grid is slow and
low-dimensional, but it is unconditionally correct: no optimizer, no Laplace approximation,
no sampler tuning. That makes it the right instrument for VERIFYING a ratio estimator --
when a grid posterior disagrees with a closed form, the ratio is wrong, and there is no
third possibility to rule out first. Sampling machinery is the natural next step, and it
should be checked against this.
"""

import numpy as np


def composite_logpost(log_ratio_fn, x_rows, grid, log_prior=None):
    """Un-normalized log posterior over ``grid`` from one group's rows.

    Parameters
    ----------
    log_ratio_fn : callable
        ``(x_rows, theta_scalar) -> (n_rows,)`` array of per-row ``log r``.
    x_rows : array
        The group's rows.
    grid : array
        1-D grid of candidate parameter values.
    log_prior : callable or None
        ``theta -> log prior``. ``None`` means flat ON THE GRID -- which is a real
        assumption, not the absence of one: a flat prior on a bounded grid is a proper
        uniform prior whose width is the grid's.

    The marginal ``p(x_i)`` in each ratio is parameter-free, so it cancels in the
    normalization and never has to be evaluated. That cancellation is the reason a ratio
    estimator is usable for inference at all.
    """
    grid = np.asarray(grid, dtype=np.float64)
    lp = np.array([float(np.sum(log_ratio_fn(x_rows, float(t)))) for t in grid])
    if log_prior is not None:
        lp = lp + np.array([float(log_prior(float(t))) for t in grid])
    return lp


def posterior_moments(grid, logpost):
    """Normalized mean and SD of a 1-D log posterior tabulated on ``grid``.

    Trapezoid over the grid, not a naive sum: with a coarse grid the sum and the integral
    differ enough to masquerade as a calibration defect, which is precisely the kind of
    false alarm a verification harness must not generate.
    """
    grid = np.asarray(grid, dtype=np.float64)
    lp = np.asarray(logpost, dtype=np.float64)
    ok = np.isfinite(lp)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    g, lp = grid[ok], lp[ok]
    p = np.exp(lp - lp.max())
    z = np.trapezoid(p, g)
    if not np.isfinite(z) or z <= 0:
        return float("nan"), float("nan")
    p = p / z
    mean = float(np.trapezoid(p * g, g))
    var = float(np.trapezoid(p * (g - mean) ** 2, g))
    return mean, float(np.sqrt(max(var, 0.0)))


def posterior_cdf_at(grid, logpost, value):
    """CDF of the tabulated posterior evaluated at ``value`` -- the PIT / SBC statistic."""
    grid = np.asarray(grid, dtype=np.float64)
    lp = np.asarray(logpost, dtype=np.float64)
    ok = np.isfinite(lp)
    if ok.sum() < 3:
        return float("nan")
    g, lp = grid[ok], lp[ok]
    p = np.exp(lp - lp.max())
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(g))])
    if c[-1] <= 0:
        return float("nan")
    return float(np.interp(value, g, c / c[-1]))


def hierarchical_logpost_global(log_ratio_fn, groups, delta_grid, nuisance_grid,
                                log_nuisance_prior):
    """Log posterior for a GLOBAL parameter shared across groups, nuisances marginalized.

    ``groups`` is a sequence of per-group row arrays. For each candidate ``delta`` the
    per-group nuisance is integrated out on ``nuisance_grid``, and the group log-evidences
    are summed:

        log P(delta) = sum_g log integral d(theta_g) prior(theta_g)
                                 exp( sum_i log r(x[g,i], delta + theta_g) )

    MARGINALIZING rather than profiling matters. Profiling (maximizing over ``theta_g``)
    is cheaper and is what a naive multi-start fit does, but it does not account for the
    nuisance's own width, and the resulting global interval is too narrow -- by more, not
    less, as the number of groups grows, because the per-group bias is systematic and
    stacks coherently while the statistical error shrinks. That is the mechanism by which
    a hierarchical fit can get *more* confidently wrong with *more* data.
    """
    delta_grid = np.asarray(delta_grid, dtype=np.float64)
    nuis = np.asarray(nuisance_grid, dtype=np.float64)
    log_pri = np.array([float(log_nuisance_prior(float(t))) for t in nuis])

    out = np.zeros_like(delta_grid)
    for x_rows in groups:
        for k, d in enumerate(delta_grid):
            ll = np.array([float(np.sum(log_ratio_fn(x_rows, float(d) + float(t))))
                           for t in nuis]) + log_pri
            m = ll.max()
            if not np.isfinite(m):
                out[k] += -np.inf
                continue
            out[k] += m + np.log(np.trapezoid(np.exp(ll - m), nuis))
    return out
