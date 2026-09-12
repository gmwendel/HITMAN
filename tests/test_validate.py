"""The validation battery: gates measure, the declared policy judges.

The tests that matter here are the DISCRIMINATING ones -- a gate that only ever passes is
not a gate. Each block below is checked against both a healthy model and a deliberately
broken one, and the policy is checked to actually flag the broken case.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.diagnostics.calibration import roc_auc
from hitman.ratio import (
    RatioBundle, build_ratio_model, compute_support, fit_ratio, save_bundle,
)
from hitman.validate import (
    RATIO_POLICY, ValidationReceipt, calibration_block, closure_block, coverage_block,
    ess_block, ood_block, run_validation,
)
from test_ratio_lane import make_grouped_dataset


@pytest.fixture(scope="module")
def trained():
    ds = make_grouped_dataset(n_groups=300, rows_per_group=10, seed=11)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=32, depth=2)
    res = fit_ratio(model, ds, key=jax.random.PRNGKey(1), steps=300, batch_size=256,
                    learning_rate=3e-3, val_every=100, residency="host", verbose=False)
    bundle = RatioBundle(model=res.model, support=compute_support(res.model, ds),
                         train_receipt=res.to_dict())
    return bundle, ds


# ---------------------------------------------------------------------------
# roc_auc primitive (moved into diagnostics; the loop depends on it)
# ---------------------------------------------------------------------------
def test_roc_auc_known_values():
    assert roc_auc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert roc_auc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)
    assert roc_auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)  # all ties -> midranks
    assert np.isnan(roc_auc([], [1.0]))


def test_roc_auc_matches_a_brute_force_pair_count():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.7, 1, 300), rng.normal(0, 1, 250)
    brute = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert roc_auc(pos, neg) == pytest.approx(brute, abs=1e-9)


# ---------------------------------------------------------------------------
# V1 / V2
# ---------------------------------------------------------------------------
def test_calibration_block_on_a_trained_model(trained):
    bundle, ds = trained
    b = calibration_block(bundle, ds, key=jax.random.PRNGKey(0))
    assert b["status"] == "ok"
    assert b["auc"] > 0.6
    assert 0.0 <= b["ece"] <= 1.0
    assert np.isfinite(b["self_norm"]) and np.isfinite(b["temperature"])


def test_calibration_block_detects_an_injected_miscalibration(trained):
    """Scaling the logits by 3 makes the classifier overconfident; ECE and the fitted
    temperature must both move, and the policy must flag it."""
    import equinox as eqx

    bundle, ds = trained
    healthy = calibration_block(bundle, ds, key=jax.random.PRNGKey(0))

    hot = eqx.tree_at(lambda m: m.net.layers[-1].weight, bundle.model,
                      bundle.model.net.layers[-1].weight * 6.0)
    broken = calibration_block(RatioBundle(model=hot, support=bundle.support), ds,
                               key=jax.random.PRNGKey(0))
    assert broken["ece"] > healthy["ece"]
    assert abs(broken["temperature"] - 1.0) > abs(healthy["temperature"] - 1.0)


def test_calibration_block_reports_empty_split_rather_than_crashing(trained):
    bundle, ds = trained
    from hitman.ratio import RatioDataset
    no_test = RatioDataset(**{**ds.__dict__,
                              "split": np.where(ds.split == "test", "train", ds.split)})
    assert calibration_block(bundle, no_test, key=jax.random.PRNGKey(0))["status"] \
        == "empty_split"


# ---------------------------------------------------------------------------
# V3 coverage
# ---------------------------------------------------------------------------
def test_coverage_block_produces_pit_values_and_a_ks_distance(trained):
    bundle, ds = trained
    b = coverage_block(bundle, ds, dim=0, n_groups=40, n_grid=25)
    assert b["status"] == "ok"
    assert b["theta_name"] == "hyp_u"
    assert 0.0 <= b["ks_uniform"] <= 1.0
    assert 0.0 <= b["pit_mean"] <= 1.0
    assert len(b["levels"]) == len(b["coverage"])


def test_coverage_ks_is_small_for_a_uniform_pit_and_large_for_a_skewed_one():
    """Calibrates the gate itself: the KS statistic must actually separate."""
    rng = np.random.default_rng(0)
    n = 200
    uni = np.sort(rng.uniform(size=n))
    ks_uni = np.max(np.abs(uni - (np.arange(1, n + 1) - 0.5) / n))
    skew = np.sort(rng.beta(2.0, 5.0, size=n))
    ks_skew = np.max(np.abs(skew - (np.arange(1, n + 1) - 0.5) / n))
    assert ks_uni < 0.12
    assert ks_skew > 0.15
    assert ks_skew > ks_uni


def test_coverage_block_on_an_uninformative_theta_dim_is_still_well_formed(trained):
    """dim 1 of the synthetic problem does not influence x at all, so the posterior is
    flat -- the gate must return a sane block rather than a nan or an exception."""
    bundle, ds = trained
    b = coverage_block(bundle, ds, dim=1, n_groups=25, n_grid=21)
    assert b["status"] == "ok" and np.isfinite(b["ks_uniform"])


# ---------------------------------------------------------------------------
# V4 closure
# ---------------------------------------------------------------------------
def test_closure_block_on_honest_and_dishonest_uncertainties():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=500)
    sigma = np.full(500, 0.5)

    honest = closure_block(truth + rng.normal(0, 0.5, 500), truth, sigma)
    assert honest["pull_sd"] == pytest.approx(1.0, abs=0.15)
    assert honest["pull_median"] == pytest.approx(0.0, abs=0.15)

    # sigma understated by 3x -> pull width ~3 (the anti-conservative-Laplace signature)
    over = closure_block(truth + rng.normal(0, 1.5, 500), truth, sigma)
    assert over["pull_sd"] > 2.0
    assert over["pull_robust_sd"] > 2.0

    biased = closure_block(truth + 1.0 + rng.normal(0, 0.5, 500), truth, sigma)
    assert abs(biased["pull_median"]) > 1.0


def test_closure_block_is_robust_to_a_few_failed_fits():
    """A handful of blown-up fits must not swamp the bulk statistics."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=300)
    est = truth + rng.normal(0, 0.5, 300)
    est[:5] = 1e6
    b = closure_block(est, truth, np.full(300, 0.5))
    assert b["pull_robust_sd"] == pytest.approx(1.0, abs=0.2)
    assert b["pull_sd"] > 100.0  # the non-robust moment IS destroyed, as expected


def test_closure_block_reports_when_nothing_is_fittable():
    b = closure_block([np.nan, np.nan], [0.0, 0.0], [0.0, 0.0])
    assert b["status"] == "no_valid_fits"


# ---------------------------------------------------------------------------
# V5 OOD + ESS
# ---------------------------------------------------------------------------
def test_ood_block_is_small_in_distribution_and_large_out(trained):
    bundle, ds = trained
    inside = ood_block(bundle, ds)
    assert inside["status"] == "ok"
    assert inside["frac_out_of_distribution"] < 0.05

    from hitman.ratio import RatioDataset
    shifted = RatioDataset(**{**ds.__dict__, "theta": ds.theta + 25.0})
    outside = ood_block(bundle, shifted)
    assert outside["frac_out_of_distribution"] > 0.9


def test_ood_block_without_support_says_so(trained):
    bundle, ds = trained
    assert ood_block(RatioBundle(model=bundle.model), ds)["status"] == "no_support_recorded"


def test_ess_block_reports_per_source(trained):
    _bundle, ds = trained
    b = ess_block(ds)
    assert b["status"] == "ok"
    assert b["ess_overall"] == pytest.approx(1.0)
    assert b["min_source_ess"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the receipt
# ---------------------------------------------------------------------------
def test_run_validation_emits_a_receipt_with_a_top_level_status(trained, tmp_path):
    bundle, ds = trained
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    closure = (truth + rng.normal(0, 0.5, 200), truth, np.full(200, 0.5))
    receipt = run_validation(bundle, ds, key=jax.random.PRNGKey(0), closure=closure,
                             coverage_dim=0)
    assert isinstance(receipt, ValidationReceipt)
    assert receipt.status in ("pass", "warn", "fail")
    for gate in ("calibration", "coverage", "ood", "ess", "closure"):
        assert gate in receipt.blocks
        assert receipt.blocks[gate]["status"] == "ok"
    assert receipt.findings, "policy produced no findings at all -- gates not wired"

    path = receipt.write(str(tmp_path / "validation.json"))
    with open(path) as f:
        d = json.load(f)
    assert d["status"] == receipt.status
    assert "validate" in d["testpoints"]


def test_policy_flags_a_broken_closure(trained):
    """The battery must actually FAIL something that is wrong."""
    bundle, ds = trained
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    # sigma understated 5x -> pull width ~5, far outside the declared band
    bad = (truth + rng.normal(0, 2.5, 200), truth, np.full(200, 0.5))
    receipt = run_validation(bundle, ds, key=jax.random.PRNGKey(0), closure=bad)
    assert receipt.status == "fail"
    assert receipt.failed
    metrics = {f["metric"] for f in receipt.findings if f["status"] == "fail"}
    assert any("pull_robust_sd" in m for m in metrics), metrics


def test_a_broken_gate_does_not_abort_the_battery(trained):
    """One diagnostic failing must never cost the other four."""
    bundle, ds = trained
    receipt = run_validation(bundle, ds, key=jax.random.PRNGKey(0),
                             coverage_dim=99)   # out of range -> the gate raises
    assert receipt.blocks["coverage"]["status"] == "error"
    assert receipt.blocks["calibration"]["status"] == "ok"
    assert receipt.blocks["ood"]["status"] == "ok"


def test_skipped_gates_are_recorded_as_skipped_not_passed(trained):
    bundle, ds = trained
    receipt = run_validation(bundle, ds, key=jax.random.PRNGKey(0), skip=("coverage",))
    assert receipt.blocks["coverage"]["status"] == "skipped"
    assert receipt.blocks["closure"]["status"] == "not_supplied"


def test_policy_thresholds_are_declared_in_code():
    """The verdict must be attributable to a commit, not to a judgement call."""
    paths = {t.path for t in RATIO_POLICY.thresholds}
    for expected in ("calibration.ece", "calibration.self_norm", "coverage.ks_uniform",
                     "closure.pull_robust_sd", "ood.frac_out_of_distribution",
                     "ess.min_source_ess"):
        assert expected in paths, f"{expected} is not gated"
    for t in RATIO_POLICY.thresholds:
        assert t.warn <= t.fail, f"{t.path}: warn must not exceed fail"
