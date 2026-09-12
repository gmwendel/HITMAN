"""The validation chain: calibration, coverage, closure and OOD as declared gates.

Not a notebook of plots -- a battery that runs at the end of every fit, judges itself
against thresholds that live in committed code, and emits a receipt whose top-level
``status`` is the answer. See :mod:`hitman.validate.report`.

    receipt = run_validation(bundle, dataset, key=k)
    receipt.write("run/validation.json")
    assert not receipt.failed
"""

from hitman.validate.gates import (
    calibration_block, closure_block, coverage_block, ess_block, ood_block, ratio_pit_1d,
)
from hitman.validate.report import (
    RATIO_POLICY, RECEIPT_SCHEMA, ValidationReceipt, run_validation,
)

__all__ = [
    "calibration_block", "coverage_block", "closure_block", "ood_block", "ess_block",
    "ratio_pit_1d", "run_validation", "ValidationReceipt", "RATIO_POLICY",
    "RECEIPT_SCHEMA",
]
