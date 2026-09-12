"""receipts.json schema: assembly, (de)serialization, and structural validation.

A receipts.json is the machine-readable output of ``python -m hitman.receipts``. It is
flat, versioned, and self-describing so the regression layer (:mod:`hitman.receipts.thresholds`)
can diff two runs mechanically.

Top-level structure::

    {
      "schema_version": "1.0",
      "meta": {model_dir, git_sha, timestamp, hitman_version, config, testpoint_tags},
      "testpoints": {
        "<tag>": {
          "truth": [7 floats],
          "self_norm": {"E_r": float, "ess": float},
          "forward": {
            "per_pmt_charge": <chi2 block>,
            "toa":            <chi2 block>,
            "n_tot":          <chi2 block + mean_sur/mean_mc>,
            "per_ring_time":  {"global": <chi2 block>, "rings": [<chi2 block>...]}
          },
          "score_identity": {"components": {name: {mean,sem,sigma}}, "max_sigma": float, "n": int},
          "mle": {"per_param": {name: {bias,resolution,pull_sigma}}, "psi_median_deg", "n_events", "mean_nhit"},
          "nuts": {"rhat_med", "rhat_p95", "divergent_frac", ...}   # optional
        }
      }
    }
"""

import json
from typing import Optional

import numpy as np

SCHEMA_VERSION = "1.0"


def _f(x) -> float:
    return float(np.asarray(x))


def chi2_block(res) -> dict:
    """Serialize a :class:`hitman.receipts.chi2.Chi2Result`."""
    pulls = np.asarray(res.pulls)
    return {
        "chi2": _f(res.chi2),
        "dof": int(res.dof),
        "chi2_dof": _f(res.chi2_dof),
        "mc_var_share": _f(res.mc_var_share),
        "n_selected": int(res.n_selected),
        "pull_std": _f(pulls.std()) if pulls.size else float("nan"),
        "pull_max_abs": _f(np.abs(pulls).max()) if pulls.size else float("nan"),
    }


def ntot_block(ntot) -> dict:
    """Serialize a :class:`hitman.wc.receipts.forward.NTotResult`."""
    d = chi2_block(ntot.chi2)
    d["mean_sur"] = _f(ntot.mean_sur)
    d["mean_mc"] = _f(ntot.mean_mc)
    return d


def ring_time_block(ring) -> dict:
    """Serialize a :class:`hitman.wc.receipts.forward.RingTimeResult`."""
    return {
        "global": chi2_block(ring.global_chi2),
        "rings": [chi2_block(r) for r in ring.per_ring],
    }


def score_block(si) -> dict:
    """Serialize a :class:`hitman.receipts.score.ScoreIdentity`."""
    return {
        "components": {
            c.name: {"mean": _f(c.mean), "sem": _f(c.sem), "sigma": _f(c.sigma)}
            for c in si.components
        },
        "max_sigma": _f(si.max_sigma),
        "n": int(si.n),
    }


def mle_block(m) -> dict:
    """Serialize a :class:`hitman.wc.receipts.mle.MLEResult`."""
    return {
        "per_param": {
            p.name: {
                "bias": _f(p.bias),
                "resolution": _f(p.resolution),
                "pull_sigma": _f(p.pull_sigma),
            }
            for p in m.params
        },
        "psi_median_deg": _f(m.psi_median_deg),
        "n_events": int(m.n_events),
        "mean_nhit": _f(m.mean_nhit),
    }


def testpoint_receipt(
    truth,
    self_norm: float,
    ess: float,
    forward: Optional[dict] = None,
    score_identity: Optional[dict] = None,
    mle: Optional[dict] = None,
    nuts: Optional[dict] = None,
) -> dict:
    """Assemble the per-testpoint block. ``forward`` etc. are pre-serialized dicts."""
    block = {
        "truth": [float(v) for v in np.asarray(truth)],
        "self_norm": {"E_r": float(self_norm), "ess": float(ess)},
    }
    if forward is not None:
        block["forward"] = forward
    if score_identity is not None:
        block["score_identity"] = score_identity
    if mle is not None:
        block["mle"] = mle
    if nuts is not None:
        block["nuts"] = nuts
    return block


def assemble(meta: dict, testpoints: dict) -> dict:
    """Wrap meta + per-testpoint blocks into a versioned receipts document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "testpoints": testpoints,
    }


def write_receipts(path: str, receipts: dict) -> None:
    validate_schema(receipts)
    with open(path, "w") as f:
        json.dump(receipts, f, indent=2, sort_keys=True)


def load_receipts(path: str) -> dict:
    with open(path) as f:
        receipts = json.load(f)
    validate_schema(receipts)
    return receipts


# Water-Cherenkov hypothesis dimension — the default when a document does not declare its
# own. Detector-agnostic documents may set ``meta.hyp_dim`` (e.g. from a HypSpec) to their
# own hypothesis length; the WC receipts keep the historical 7 with no change on their side.
DEFAULT_HYP_DIM = 7


def validate_schema(receipts: dict, hyp_dim: Optional[int] = None) -> None:
    """Cheap structural validation — raises ``ValueError`` on a malformed document.

    The expected ``truth`` length is, in order of precedence: the explicit ``hyp_dim``
    argument, else ``receipts["meta"]["hyp_dim"]`` if present, else
    :data:`DEFAULT_HYP_DIM` (7, water-Cherenkov). A detector with a different hypothesis
    dimension records ``meta.hyp_dim`` (see :class:`hitman.spec.HypSpec`) and validates
    with no other change.
    """
    if receipts.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {receipts.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in ("meta", "testpoints"):
        if key not in receipts:
            raise ValueError(f"missing top-level key {key!r}")
    if not isinstance(receipts["testpoints"], dict):
        raise ValueError("'testpoints' must be a mapping tag -> block")
    if hyp_dim is None:
        hyp_dim = receipts.get("meta", {}).get("hyp_dim", DEFAULT_HYP_DIM)
    for tag, block in receipts["testpoints"].items():
        for req in ("truth", "self_norm"):
            if req not in block:
                raise ValueError(f"testpoint {tag!r} missing {req!r}")
        if len(block["truth"]) != hyp_dim:
            raise ValueError(
                f"testpoint {tag!r} truth must have {hyp_dim} entries")
