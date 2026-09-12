"""Validation receipts: run the gates, judge them against a DECLARED policy, emit JSON.

The point of this module is that "did the model pass?" is a FIELD, not a reading exercise.
A training run's artifact is a receipt with a top-level status, and the thresholds that
produced it are committed code rather than a judgement made while looking at a plot.

Judging reuses :mod:`hitman.receipts.thresholds` -- the same ``Threshold`` / ``Finding`` /
``worst_status`` engine the water-Cherenkov battery uses -- rather than a second verdict
implementation. The receipt is shaped as ``{"testpoints": {block: {...}}}`` because that
is the shape that engine reads.

The default thresholds below are STARTING POINTS chosen from what the metrics mean, not
from measurements on this program's data. They are deliberately loose enough not to cry
wolf and tight enough to catch a real regression; expect to tighten them once a few runs
have established the achievable band. They are versioned with the code so a change of
verdict is always attributable to a commit.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from hitman.receipts.thresholds import (
    DEFAULT_POLICY as _WC_POLICY, Policy, Threshold, evaluate, summary,
)

RECEIPT_SCHEMA = "hitman-validate-receipt-v1"

#: Absolute gates for the ratio lane. See the module docstring on provenance.
RATIO_POLICY = Policy(thresholds=[
    # V1 -- classifier calibration. ECE is a probability, so these are absolute.
    Threshold("calibration.ece", 0.02, 0.05),
    # V2 -- ratio self-normalization: E[r] over marginal pairs is exactly 1 when optimal.
    Threshold("calibration.self_norm", 0.10, 0.30, mode="abs_dev_from", target=1.0),
    # ...and the temperature that would fix it. tau far from 1 means overconfident logits.
    Threshold("calibration.temperature", 0.15, 0.40, mode="abs_dev_from", target=1.0),
    # V3 -- posterior coverage. KS distance of the PIT sample from Uniform(0,1).
    Threshold("coverage.ks_uniform", 0.15, 0.25),
    Threshold("coverage.max_coverage_deviation", 0.10, 0.20),
    # V4 -- closure. Pull width 1 is the honest value; bias is checked separately.
    Threshold("closure.pull_robust_sd", 0.25, 0.50, mode="abs_dev_from", target=1.0),
    Threshold("closure.pull_median", 0.20, 0.50, mode="abs_dev_from", target=0.0),
    # V5 -- operating-point support occupancy and reweighting health.
    Threshold("ood.frac_out_of_distribution", 0.05, 0.20),
    Threshold("ess.min_source_ess", 0.70, 0.90, mode="abs_dev_from", target=1.0),
])


@dataclass
class ValidationReceipt:
    """The artifact. ``status`` is the one field a caller has to read."""

    schema: str = RECEIPT_SCHEMA
    status: str = "pass"
    blocks: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "status": self.status, "meta": self.meta,
            "counts": self.counts,
            "findings": [asdict(f) if not isinstance(f, dict) else f
                         for f in self.findings],
            "testpoints": {"validate": self.blocks},
        }

    def write(self, path: str) -> str:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        os.replace(tmp, path)
        return path

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def run_validation(bundle, dataset, *, key, closure=None, policy: Policy = RATIO_POLICY,
                   coverage_dim: int = 0, split: str = "test",
                   baseline: Optional[dict] = None, meta: Optional[dict] = None,
                   skip=()) -> ValidationReceipt:
    """Run every gate over ``(bundle, dataset)`` and judge against ``policy``.

    ``closure`` is an optional ``(estimates, truths, sigmas)`` triple from an end-to-end
    study; the closure gate is skipped when it is absent rather than fabricated, because a
    closure test needs an inference procedure and this module deliberately does not own
    one.

    A gate that raises is recorded as ``status: "error"`` with its message and does not
    abort the rest -- one broken diagnostic must never cost you the other four. An errored
    gate contributes no findings, so it also cannot silently turn a fail into a pass: the
    receipt shows it plainly.
    """
    from hitman.validate import gates as G

    blocks: dict = {}

    def _run(name, fn):
        if name in skip:
            blocks[name] = {"status": "skipped"}
            return
        try:
            blocks[name] = fn()
        except Exception as exc:  # noqa: BLE001 - one gate must not kill the battery
            blocks[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    _run("calibration", lambda: G.calibration_block(bundle, dataset, key=key, split=split))
    _run("coverage", lambda: G.coverage_block(bundle, dataset, dim=coverage_dim,
                                              split=split))
    _run("ood", lambda: G.ood_block(bundle, dataset, split=split))
    _run("ess", lambda: G.ess_block(dataset))
    if closure is not None:
        est, tru, sig = closure
        _run("closure", lambda: G.closure_block(est, tru, sig))
    else:
        blocks["closure"] = {"status": "not_supplied"}

    receipt_dict = {"testpoints": {"validate": blocks}}
    findings = evaluate(receipt_dict, policy, baseline)
    summ = summary(findings)
    return ValidationReceipt(
        status=summ["worst"], blocks=blocks,
        findings=[asdict(f) for f in findings], counts=summ["counts"],
        meta={**(meta or {}), "dataset": dataset.describe(),
              "bundle": bundle.describe()},
    )
