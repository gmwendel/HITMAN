"""The validation gates: calibration, coverage, closure, out-of-distribution.

Each gate takes a ``(bundle, dataset)`` and returns a flat, JSON-able block of scalars.
:mod:`hitman.validate.report` turns those blocks into pass/warn/fail findings against a
declared policy. Nothing here decides a verdict -- gates measure, the policy judges.

Why these four, in this order:

V1/V2 (calibration) are properties of the CLASSIFIER and cost one forward pass: is the
logit a calibrated log-ratio at all? A miscalibrated classifier makes everything
downstream meaningless, so it is checked first and it is cheap.

V3 (coverage) is a property of the POSTERIOR the ratio induces. It is the gate that
actually answers "are my error bars honest", and it is the one a BCE curve can never
answer -- a well-separated classifier can still induce badly-scaled intervals.

V4 (closure) is end-to-end: inject a known value, recover it, check the pull.

V5 (OOD) is a property of the OPERATING POINT rather than the model: how much of what you
are about to evaluate lies outside where the model was trained. Reported, never silently
corrected -- see :func:`hitman.ratio.support.trust_radius`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hitman.diagnostics.calibration import (
    expected_calibration_error, expected_coverage, fit_temperature, reliability_curve,
    roc_auc, self_normalization,
)
from hitman.ratio.pairing import build_group_index, get_pairing


# ---------------------------------------------------------------------------
# V1 / V2 -- classifier calibration and ratio self-normalization
# ---------------------------------------------------------------------------
def calibration_block(bundle, dataset, *, key, split: str = "test",
                      pairing="group_shift", max_rows: int = 200_000,
                      n_bins: int = 15) -> dict:
    """Reliability/ECE, AUC, ratio self-normalization and the fitted temperature.

    ``self_normalization`` is the sharpest of these: ``E_{p(x)p(theta)}[r] = 1`` holds
    exactly for the optimal classifier, so a departure is direct evidence of a
    mis-normalized -- typically overconfident -- ratio, independent of how well it
    separates.

    ``temperature`` recalibrates as ``logit / tau``; ``tau ~ 1`` is calibrated, ``tau > 1``
    overconfident. It is REPORTED, not applied: silently rescaling a trained model's logit
    would change every downstream fit.
    """
    sub = dataset.split_view(split)
    if sub.n_rows == 0:
        return {"status": "empty_split", "split": split}
    bundle.assert_compatible(sub)

    rows = np.arange(sub.n_rows)
    if max_rows and sub.n_rows > max_rows:
        rows = np.sort(np.random.default_rng(0).choice(sub.n_rows, max_rows, replace=False))
    index = build_group_index(sub.group_id)
    pair = get_pairing(pairing)
    rows_j = jnp.asarray(rows.astype(np.int32))
    neg = (pair(key, rows_j, index.to_device()) if pair.jittable
           else pair(np.random.default_rng(0), rows, index))

    x = jnp.asarray(sub.x)[rows_j]
    th_pos = jnp.asarray(sub.theta)[rows_j]
    th_neg = jnp.asarray(sub.theta)[jnp.asarray(np.asarray(neg))]
    lp = np.asarray(bundle.model.logit_batch(x, th_pos))
    ln = np.asarray(bundle.model.logit_batch(x, th_neg))

    _conf, _acc, ece = reliability_curve(lp, ln, n_bins)
    return {
        "status": "ok", "split": split, "n_rows": int(rows.size),
        "ece": float(ece),
        "auc": float(roc_auc(lp, ln)),
        "self_norm": float(self_normalization(ln)),
        "temperature": float(fit_temperature(lp, ln)),
        "mean_logit_joint": float(np.mean(lp)),
        "mean_logit_marginal": float(np.mean(ln)),
    }


# ---------------------------------------------------------------------------
# V3 -- coverage of the ratio-induced 1-D posterior
# ---------------------------------------------------------------------------
def _group_slices(group_id):
    order = np.argsort(group_id, kind="stable")
    g = np.asarray(group_id)[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], g.size)
    return order, uniq, start, end


def ratio_pit_1d(model, x_rows, theta_row, dim: int, grid: np.ndarray) -> float:
    """Probability-integral transform of the truth under the ratio's 1-D posterior.

    For one group, hold every hypothesis component at truth except ``dim``, sweep ``dim``
    over ``grid``, and form ``log P(theta_d) = sum_i log r(x_i | theta)`` (a flat prior on
    the grid). Normalizing over the grid gives a proper 1-D posterior; the returned value
    is its CDF evaluated at the true ``theta_d``.

    If the posterior is calibrated these PIT values are Uniform(0, 1) across groups --
    the continuous form of an SBC rank, without needing posterior SAMPLES, which a ratio
    estimator does not directly provide.

    The flat-prior-on-the-grid choice is deliberate and is a REAL assumption: the learned
    ratio's denominator is the training marginal, so this posterior is implicitly
    referenced to the training proposal. Comparing PITs across models trained on different
    proposals is not meaningful.
    """
    thetas = np.repeat(np.asarray(theta_row)[None, :], len(grid), axis=0)
    thetas[:, dim] = grid
    xs = jnp.asarray(x_rows)
    logp = np.array([
        float(jnp.sum(model.logit_batch(xs, jnp.repeat(jnp.asarray(t)[None, :],
                                                       xs.shape[0], axis=0))))
        for t in thetas
    ])
    logp -= logp.max()
    p = np.exp(logp)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    cdf = np.cumsum(p) / total
    truth = float(np.asarray(theta_row)[dim])
    return float(np.interp(truth, grid, cdf))


def coverage_block(bundle, dataset, *, dim: int = 0, split: str = "test",
                   n_groups: int = 64, n_grid: int = 33, seed: int = 0,
                   levels=None) -> dict:
    """PIT uniformity + empirical coverage of central intervals for one theta dimension."""
    sub = dataset.split_view(split)
    if sub.n_rows == 0:
        return {"status": "empty_split", "split": split}
    bundle.assert_compatible(sub)

    order, uniq, start, end = _group_slices(sub.group_id)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(uniq), size=min(n_groups, len(uniq)), replace=False)
    col = np.asarray(sub.theta)[:, dim]
    grid = np.linspace(float(col.min()), float(col.max()), int(n_grid))
    if not np.isfinite(grid).all() or grid[0] == grid[-1]:
        return {"status": "degenerate_grid", "dim": int(dim)}

    pits = []
    for gi in pick:
        rows = order[start[gi]:end[gi]]
        pits.append(ratio_pit_1d(bundle.model, np.asarray(sub.x)[rows],
                                 np.asarray(sub.theta)[rows[0]], dim, grid))
    pits = np.asarray([p for p in pits if np.isfinite(p)], dtype=np.float64)
    if pits.size == 0:
        return {"status": "no_finite_pit", "dim": int(dim)}

    lv, cov = expected_coverage(pits[:, None], n_draws=1, levels=levels)
    # KS distance of the PIT sample from Uniform(0,1) -- the scalar the policy gates on
    s = np.sort(pits)
    n = s.size
    ks = float(np.max(np.abs(s - (np.arange(1, n + 1) - 0.5) / n)))
    return {
        "status": "ok", "split": split, "dim": int(dim),
        "theta_name": (dataset.theta_names[dim] if dataset.theta_names else str(dim)),
        "n_groups": int(n), "ks_uniform": ks,
        "pit_mean": float(pits.mean()), "pit_std": float(pits.std()),
        "levels": np.asarray(lv).tolist(),
        "coverage": np.asarray(cov).ravel().tolist(),
        "max_coverage_deviation": float(np.max(np.abs(np.asarray(cov).ravel()
                                                      - np.asarray(lv)))),
    }


# ---------------------------------------------------------------------------
# V4 -- closure on an injected scalar
# ---------------------------------------------------------------------------
def closure_block(estimates, truths, sigmas, *, name: str = "parameter") -> dict:
    """Pull statistics for an injected-truth closure test.

    ``pull = (estimate - truth) / sigma`` should be mean 0, SD 1 if the point estimate is
    unbiased AND the reported uncertainty is honest. Robust (median / MAD-based) versions
    are reported alongside, because a handful of failed fits otherwise dominate the moments
    and hide the behaviour of the bulk.
    """
    est = np.asarray(estimates, dtype=np.float64)
    tru = np.asarray(truths, dtype=np.float64)
    sig = np.asarray(sigmas, dtype=np.float64)
    resid = est - tru
    ok = np.isfinite(resid) & np.isfinite(sig) & (sig > 0)
    if not np.any(ok):
        return {"status": "no_valid_fits", "name": name}
    pull = resid[ok] / sig[ok]

    def robust_sd(a):
        return float(1.4826 * np.median(np.abs(a - np.median(a))))

    return {
        "status": "ok", "name": name, "n": int(ok.sum()),
        "n_invalid": int((~ok).sum()),
        "resid_mean": float(resid[ok].mean()), "resid_rms": float(np.sqrt(np.mean(resid[ok] ** 2))),
        "resid_robust_sd": robust_sd(resid[ok]),
        "pull_mean": float(pull.mean()), "pull_sd": float(pull.std()),
        "pull_median": float(np.median(pull)), "pull_robust_sd": robust_sd(pull),
        "abs_pull_median": float(np.median(np.abs(pull))),
    }


# ---------------------------------------------------------------------------
# V5 -- out-of-distribution occupancy
# ---------------------------------------------------------------------------
def ood_block(bundle, dataset, *, split: str = "test", prob: float = 0.999,
              max_rows: int = 200_000, seed: int = 0) -> dict:
    """How much of ``split`` lies outside the model's recorded training support.

    Reported per source as well as overall: a combined dataset can look fine while one
    contributing pool sits almost entirely outside the support, which is exactly the
    situation reweighting is asked to paper over and cannot.
    """
    support = bundle.support
    if support is None:
        return {"status": "no_support_recorded"}
    sub = dataset.split_view(split)
    if sub.n_rows == 0:
        return {"status": "empty_split", "split": split}
    bundle.assert_compatible(sub)

    rows = np.arange(sub.n_rows)
    if max_rows and sub.n_rows > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(sub.n_rows, max_rows, replace=False))
    feat = np.asarray(jax.vmap(bundle.model.standardized)(
        jnp.asarray(sub.x)[rows], jnp.asarray(sub.theta)[rows]))
    d = support.d_mahalanobis(feat)
    thresh = support.dm_at(prob)
    sid = np.asarray(sub.source_id)[rows]
    return {
        "status": "ok", "split": split, "n_rows": int(rows.size),
        "threshold": float(thresh), "threshold_prob": float(prob),
        "frac_out_of_distribution": float(np.mean(d > thresh)),
        "d_median": float(np.median(d)), "d_p99": float(np.quantile(d, 0.99)),
        "d_max": float(d.max()),
        "per_source_frac_ood": {int(s): float(np.mean(d[sid == s] > thresh))
                                for s in np.unique(sid)},
    }


def ess_block(dataset) -> dict:
    """Effective sample size overall and per source -- the reweighting health metric."""
    from hitman.ratio.dataset import effective_sample_size

    ess = effective_sample_size(dataset.weight, dataset.source_id)
    per = ess.get("per_source", {})
    return {
        "status": "ok", "ess_overall": ess["overall"],
        "per_source_ess": {int(k): float(v) for k, v in per.items()},
        "min_source_ess": float(min(per.values())) if per else ess["overall"],
    }
