"""Threshold policy: turn a receipts.json into pass / warn / fail findings.

Two judgement modes coexist:

* **Absolute gates** — physics-motivated ceilings that hold for any run
  (chi2/dof, self-normalization, score sigma, pull width). Always active.
* **Regression gates** — baseline-relative: flag a metric that *worsened* versus a
  recorded baseline receipts.json by more than a tolerance. Active only when a baseline
  is supplied. This is how L3 catches a silent degradation whose absolute value is still
  under the ceiling.

A missing metric (e.g. optional NUTS block) is skipped, not failed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Status ordering for taking the worst.
PASS, WARN, FAIL = "pass", "warn", "fail"
_ORDER = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass(frozen=True)
class Threshold:
    """One rule applied to a dotted metric path inside a testpoint block.

    mode
        ``"max"``            : value must be <= warn/fail.
        ``"abs_dev_from"``   : ``|value - target|`` must be <= warn/fail.
        ``"baseline_ratio"`` : ``value / baseline`` must be <= warn/fail (regression).
        ``"baseline_delta"`` : ``value - baseline`` must be <= warn/fail (regression).
    for_each
        If given, ``path`` is a template with ``{p}`` expanded over these names
        (e.g. per-parameter pull widths).
    """

    path: str
    warn: float
    fail: float
    mode: str = "max"
    target: float = 1.0
    for_each: Optional[Sequence[str]] = None

    @property
    def is_regression(self) -> bool:
        return self.mode.startswith("baseline")


@dataclass
class Finding:
    testpoint: str
    metric: str
    value: float
    status: str
    mode: str
    detail: str = ""


@dataclass
class Policy:
    thresholds: List[Threshold] = field(default_factory=list)


# ---- default policy ----------------------------------------------------------

DEFAULT_POLICY = Policy(
    thresholds=[
        # absolute forward-model gates
        Threshold("forward.per_pmt_charge.chi2_dof", 2.0, 5.0),
        Threshold("forward.toa.chi2_dof", 2.0, 5.0),
        Threshold("forward.n_tot.chi2_dof", 2.0, 5.0),
        Threshold("forward.per_ring_time.global.chi2_dof", 2.0, 5.0),
        # normalization / calibration
        Threshold("self_norm.E_r", 0.1, 0.3, mode="abs_dev_from", target=1.0),
        # inference-layer gates
        Threshold("score_identity.max_sigma", 3.0, 5.0),
        Threshold(
            "mle.per_param.{p}.pull_sigma",
            0.3,
            0.6,
            mode="abs_dev_from",
            target=1.0,
            for_each=("x", "z", "t", "E"),
        ),
        # optional NUTS gates (skipped if the block is absent)
        Threshold("nuts.rhat_p95", 1.05, 1.2),
        # regression gates (only fire when a baseline is supplied)
        Threshold("forward.per_pmt_charge.chi2_dof", 1.5, 2.5, mode="baseline_ratio"),
        Threshold("forward.per_ring_time.global.chi2_dof", 1.5, 2.5, mode="baseline_ratio"),
        Threshold("mle.psi_median_deg", 1.15, 1.4, mode="baseline_ratio"),
        Threshold(
            "mle.per_param.{p}.resolution",
            1.15,
            1.4,
            mode="baseline_ratio",
            for_each=("x", "z", "t", "E"),
        ),
    ]
)


def _get(block: dict, dotted: str):
    node = block
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _judge(value: float, base: Optional[float], th: Threshold):
    """Return (status, score, detail) for one metric value under one threshold."""
    if th.mode == "max":
        score = value
        detail = f"{value:.4g} (warn>{th.warn:g}, fail>{th.fail:g})"
    elif th.mode == "abs_dev_from":
        score = abs(value - th.target)
        detail = f"|{value:.4g}-{th.target:g}|={score:.4g} (warn>{th.warn:g}, fail>{th.fail:g})"
    elif th.mode in ("baseline_ratio", "baseline_delta"):
        if base is None or (th.mode == "baseline_ratio" and base == 0):
            return None
        score = value / base if th.mode == "baseline_ratio" else value - base
        op = "x" if th.mode == "baseline_ratio" else "+"
        detail = f"{value:.4g} vs base {base:.4g} ({op}{score:.3g}; warn>{th.warn:g}, fail>{th.fail:g})"
    else:
        raise ValueError(f"unknown mode {th.mode!r}")
    status = FAIL if score > th.fail else WARN if score > th.warn else PASS
    return status, score, detail


def evaluate(
    receipts: dict, policy: Policy = DEFAULT_POLICY, baseline: Optional[dict] = None
) -> List[Finding]:
    """Apply ``policy`` to every testpoint; return findings for metrics that are present."""
    findings: List[Finding] = []
    base_tps = (baseline or {}).get("testpoints", {})
    for tag, block in receipts["testpoints"].items():
        base_block = base_tps.get(tag)
        for th in policy.thresholds:
            names = th.for_each or (None,)
            for p in names:
                path = th.path.format(p=p) if p is not None else th.path
                value = _get(block, path)
                if value is None:
                    continue
                base_val = _get(base_block, path) if base_block is not None else None
                judged = _judge(float(value), base_val, th)
                if judged is None:
                    continue
                status, _, detail = judged
                findings.append(
                    Finding(
                        testpoint=tag,
                        metric=path,
                        value=float(value),
                        status=status,
                        mode=th.mode,
                        detail=detail,
                    )
                )
    return findings


def worst_status(findings: Sequence[Finding]) -> str:
    return max((f.status for f in findings), key=lambda s: _ORDER[s], default=PASS)


def summary(findings: Sequence[Finding]) -> dict:
    """Counts by status plus the overall worst status (the process exit signal)."""
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for f in findings:
        counts[f.status] += 1
    return {"counts": counts, "worst": worst_status(findings)}
