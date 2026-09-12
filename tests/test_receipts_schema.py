import numpy as np
import pytest

from hitman.receipts import schema
from hitman.receipts.chi2 import chi2_comparison


def _minimal_receipt():
    res = chi2_comparison(np.zeros(5), np.ones(5), np.ones(5), np.ones(5), ddof=1)
    tp = schema.testpoint_receipt(
        truth=[0, 0, 0, 1.57, 0, 0, 3.0],
        self_norm=1.02,
        ess=1.2e5,
        forward={"toa": schema.chi2_block(res)},
    )
    return schema.assemble({"model_dir": "m"}, {"pt": tp})


def test_roundtrip_write_load(tmp_path):
    r = _minimal_receipt()
    p = tmp_path / "receipts.json"
    schema.write_receipts(str(p), r)
    loaded = schema.load_receipts(str(p))
    assert loaded["schema_version"] == schema.SCHEMA_VERSION
    assert loaded["testpoints"]["pt"]["self_norm"]["E_r"] == 1.02
    assert loaded["testpoints"]["pt"]["forward"]["toa"]["dof"] == 4


def test_chi2_block_is_json_scalar_typed():
    res = chi2_comparison(np.zeros(3), np.ones(3), np.ones(3), np.ones(3), ddof=1)
    b = schema.chi2_block(res)
    assert isinstance(b["chi2"], float) and isinstance(b["dof"], int)


def test_validate_rejects_bad_version():
    r = _minimal_receipt()
    r["schema_version"] = "0.0"
    with pytest.raises(ValueError):
        schema.validate_schema(r)


def test_validate_rejects_wrong_truth_length():
    r = _minimal_receipt()
    r["testpoints"]["pt"]["truth"] = [0, 0, 0]
    with pytest.raises(ValueError):
        schema.validate_schema(r)


def test_validate_rejects_missing_toplevel():
    with pytest.raises(ValueError):
        schema.validate_schema({"schema_version": schema.SCHEMA_VERSION, "meta": {}})
