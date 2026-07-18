from hitman.receipts import thresholds
from hitman.receipts.thresholds import FAIL, PASS, WARN, evaluate, summary


def _block(**over):
    b = {
        "truth": [0, 0, 0, 1.57, 0, 0, 3.0],
        "self_norm": {"E_r": 1.0, "ess": 1e5},
        "forward": {
            "per_pmt_charge": {"chi2_dof": 6.0},   # FAIL (>5)
            "toa": {"chi2_dof": 2.5},              # WARN (>2)
            "n_tot": {"chi2_dof": 1.0},            # PASS
            "per_ring_time": {"global": {"chi2_dof": 1.2}},  # PASS absolute
        },
        "score_identity": {"max_sigma": 2.0},      # PASS
        "mle": {
            "per_param": {
                "x": {"pull_sigma": 1.0, "resolution": 90.0},   # PASS
                "z": {"pull_sigma": 1.5, "resolution": 90.0},   # WARN (|1.5-1|=0.5)
                "t": {"pull_sigma": 1.0, "resolution": 0.7},
                "E": {"pull_sigma": 1.0, "resolution": 0.4},
            },
            "psi_median_deg": 12.0,
        },
    }
    b.update(over)
    return b


def _find(findings, metric, tp="pt"):
    return next(f for f in findings if f.metric == metric and f.testpoint == tp)


def test_absolute_gates_status():
    r = {"testpoints": {"pt": _block()}}
    f = evaluate(r)
    assert _find(f, "forward.per_pmt_charge.chi2_dof").status == FAIL
    assert _find(f, "forward.toa.chi2_dof").status == WARN
    assert _find(f, "forward.n_tot.chi2_dof").status == PASS
    assert _find(f, "score_identity.max_sigma").status == PASS
    assert _find(f, "mle.per_param.z.pull_sigma").status == WARN
    assert _find(f, "mle.per_param.x.pull_sigma").status == PASS


def test_self_norm_deviation_gate():
    r = {"testpoints": {"pt": _block(self_norm={"E_r": 1.4, "ess": 1e5})}}
    assert _find(evaluate(r), "self_norm.E_r").status == FAIL  # |1.4-1|=0.4 > 0.3


def test_missing_optional_metric_is_skipped():
    r = {"testpoints": {"pt": _block()}}
    metrics = {f.metric for f in evaluate(r)}
    assert "nuts.rhat_p95" not in metrics  # no nuts block -> not flagged


def test_regression_gate_fires_only_with_baseline():
    cur = {"testpoints": {"pt": _block()}}
    base = {"testpoints": {"pt": _block(
        forward={**_block()["forward"], "per_ring_time": {"global": {"chi2_dof": 0.5}}}
    )}}
    # baseline ring chi2_dof 0.5 -> current 1.2 -> ratio 2.4 -> WARN (regression), even
    # though 1.2 passes the absolute gate.
    f = evaluate(cur, baseline=base)
    reg = [x for x in f if x.metric == "forward.per_ring_time.global.chi2_dof"
           and x.mode == "baseline_ratio"]
    assert reg and reg[0].status == WARN
    # without a baseline, no regression finding exists
    f0 = evaluate(cur)
    assert not any(x.mode == "baseline_ratio" for x in f0)


def test_summary_worst_is_fail():
    r = {"testpoints": {"pt": _block()}}
    s = summary(evaluate(r))
    assert s["worst"] == FAIL
    assert s["counts"][FAIL] >= 1
