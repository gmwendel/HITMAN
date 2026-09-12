"""Forward-model receipts: surrogate-implied observables vs MC at a fixed hypothesis.

Library form of ``forward_check.py`` and ``forward_time_per_pmt.py``. Each function
takes already-computed pool log-weights / MC arrays and returns a :class:`Chi2Result`
(plus scalar summaries where relevant), so the whole battery is testable with tiny
random nets and synthetic pools — no ROOT file, no trained model.

Design report, Addendum 4 lesson: BCE is blind to region-reallocated probability mass.
These reweighting receipts are what actually caught a missing augmentation and an
undertrained chargenet (runs 3-7 saga), so they are the L2 core.
"""

from typing import NamedTuple

import numpy as np

from hitman.receipts.chi2 import Chi2Result, chi2_comparison
from hitman.receipts.reweight import PoolWeights, group_fractions, reweighted_density


class NTotResult(NamedTuple):
    chi2: Chi2Result
    mean_sur: float
    mean_mc: float


class RingTimeResult(NamedTuple):
    global_chi2: Chi2Result
    per_ring: list  # list of Chi2Result, one per ring


def implied_ntot_pmf(logit_c, train_n_hist, n_grid):
    """Chargenet-implied P(N_tot | theta) on an integer grid.

    ``P(N | theta) ∝ r_c(N, theta) * p_train(N)``; bins with no training support are
    dropped. Returns the normalized pmf over ``n_grid`` and its mean.
    """
    logit_c = np.asarray(logit_c, dtype=np.float64)
    train_n_hist = np.asarray(train_n_hist, dtype=np.float64)
    n_grid = np.asarray(n_grid)
    ok = train_n_hist > 0
    pmf = np.exp(logit_c - logit_c.max()) * train_n_hist
    pmf = np.where(ok, pmf, 0.0)
    total = pmf.sum()
    pmf = pmf / total if total > 0 else pmf
    mean = float((n_grid * pmf).sum())
    return pmf, mean


def per_pmt_charge_receipt(
    pool: PoolWeights, pool_pmt, n_pmts: int, mc_pmt, n_events: int, mean_ntot: float
) -> Chi2Result:
    """Per-PMT mean-PE agreement (observable 1 of forward_check).

    The surrogate per-PMT charge is ``<N_tot>_sur * frac_sur(pmt)``, where the fraction
    is the reweighted pool occupancy of that PMT; compared to the MC per-PMT mean.
    """
    mc_pmt = np.asarray(mc_pmt)
    mc_counts = np.bincount(mc_pmt, minlength=n_pmts).astype(np.float64)
    mc_mean = mc_counts / n_events
    mc_err = np.sqrt(mc_counts) / n_events

    frac, frac_err = group_fractions(pool.weights, pool_pmt, n_pmts)
    sur_mean = mean_ntot * frac
    sur_err = mean_ntot * frac_err
    return chi2_comparison(mc_mean, mc_err, sur_mean, sur_err, ddof=1)


def toa_receipt(pool: PoolWeights, pool_t, mc_t, bins=None, min_count: int = 5) -> Chi2Result:
    """Hit time-of-arrival density agreement (observable 2 of forward_check)."""
    mc_t = np.asarray(mc_t)
    if bins is None:
        bins = np.linspace(np.floor(mc_t.min()), np.percentile(mc_t, 99.9), 61)
    bins = np.asarray(bins)
    width = np.diff(bins)
    mc_h, _ = np.histogram(mc_t, bins)
    mc_dens = mc_h / mc_h.sum() / width
    mc_dens_err = np.sqrt(mc_h) / mc_h.sum() / width

    sur_dens, sur_dens_err = reweighted_density(pool_t, pool.weights, bins, renormalize=True)
    mask = mc_h > min_count
    return chi2_comparison(mc_dens, mc_dens_err, sur_dens, sur_dens_err, mask=mask, ddof=1)


def ntot_receipt(
    pmf, n_grid, train_n_hist, mc_ntot, n_events: int, binwidth: float = 2.0, min_count: int = 5
) -> NTotResult:
    """Total-charge distribution agreement (observable 3 of forward_check)."""
    n_grid = np.asarray(n_grid)
    pmf = np.asarray(pmf, dtype=np.float64)
    train_n_hist = np.asarray(train_n_hist, dtype=np.float64)
    mc_ntot = np.asarray(mc_ntot)
    ok = train_n_hist > 0

    nbins = np.arange(mc_ntot.min() - 0.5, mc_ntot.max() + 1.5, binwidth)
    mc_nh, _ = np.histogram(mc_ntot, nbins)
    mc_np = mc_nh / n_events
    mc_np_err = np.sqrt(mc_nh) / n_events
    centers = 0.5 * (nbins[1:] + nbins[:-1])

    sur_np = np.zeros(len(centers))
    sur_np_err = np.zeros(len(centers))
    for b in range(len(centers)):
        m = (n_grid >= nbins[b]) & (n_grid < nbins[b + 1])
        sur_np[b] = pmf[m].sum()
        tot = train_n_hist[m & ok].sum()
        rel = 1.0 / np.sqrt(tot) if tot > 0 else 0.0
        sur_np_err[b] = sur_np[b] * rel

    mask = mc_nh > min_count
    res = chi2_comparison(mc_np, mc_np_err, sur_np, sur_np_err, mask=mask, ddof=1)
    return NTotResult(chi2=res, mean_sur=float((n_grid * pmf).sum()), mean_mc=float(mc_ntot.mean()))


def per_ring_time_receipt(
    pool: PoolWeights, pool_t, pool_ring, mc_t, mc_ring, n_rings: int, bins, min_count: int = 5
) -> RingTimeResult:
    """Per-z-ring time-of-arrival agreement (forward_time_per_pmt).

    Joint ``p(ring, t)`` is normalized over the full 2-D space (each side divided by
    its own total hit count), so rings share one normalization — the receipt that
    catches per-region timing errors the pooled ToA hides.
    """
    mc_t = np.asarray(mc_t)
    mc_ring = np.asarray(mc_ring)
    pool_t = np.asarray(pool_t)
    pool_ring = np.asarray(pool_ring)
    bins = np.asarray(bins)
    n_mc = len(mc_t)

    per_ring = []
    tot_chi2 = 0.0
    tot_dof = 0
    tot_pulls = []
    tot_shares = []
    for r in range(n_rings):
        mc_sel = mc_ring == r
        mc_h, _ = np.histogram(mc_t[mc_sel], bins)
        mc_p = mc_h / n_mc
        mc_pe = np.sqrt(mc_h) / n_mc

        pl_sel = pool_ring == r
        sur_p = np.zeros(len(bins) - 1)
        sur_pe = np.zeros(len(bins) - 1)
        for b in range(len(bins) - 1):
            member = pl_sel & (pool_t >= bins[b]) & (pool_t < bins[b + 1])
            from hitman.receipts.reweight import weighted_fraction

            sur_p[b], sur_pe[b] = weighted_fraction(pool.weights, member)

        mask = mc_h > min_count
        res = chi2_comparison(mc_p, mc_pe, sur_p, sur_pe, mask=mask, ddof=0)
        per_ring.append(res)
        tot_chi2 += res.chi2
        tot_dof += res.n_selected
        tot_pulls.append(res.pulls)
        tot_shares.append(res.mc_var_share)

    dof = max(tot_dof, 1)
    global_res = Chi2Result(
        chi2=tot_chi2,
        dof=dof,
        chi2_dof=tot_chi2 / dof,
        mc_var_share=float(np.nanmedian(tot_shares)) if tot_shares else float("nan"),
        pulls=np.concatenate(tot_pulls) if tot_pulls else np.array([]),
        n_selected=tot_dof,
    )
    return RingTimeResult(global_chi2=global_res, per_ring=per_ring)
