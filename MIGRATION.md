# Migration notes — the library/application split and the ratio lane

Supersedes the previous contents of this file (the "composable SBI API", proposals 1–5).
Those are declared dead below, with the reason, rather than deleted silently.

---

## What changed

HITMAN was one water-Cherenkov application with a small library inside it. It is now a
detector-agnostic library with the water-Cherenkov application beside it, plus the two lanes
the library was missing.

```
hitman/            the library -- must never import from hitman.wc
  spec.py          observation/hypothesis layout protocol
  data/            grouped batch container + input pipeline
  nn/mlp.py        the MLP building block
  density/         exactly-normalized density factors
  ratio/           NEW -- the neural-ratio lane
  npe/             neural posterior estimation (DeepSets encoder + spline flow)
  validate/        NEW -- calibration / coverage / closure / OOD gates
  diagnostics/     the numeric primitives those gates use
  receipts/        detector-agnostic receipt core (chi2, IS weights, score, thresholds)
  calibrate/       Godambe/sandwich adjustment

hitman/wc/         the water-Cherenkov application
  inference/       the four deployment lanes (batched / seq / compiled / mle)
  splinemle/       closed-form conditional density (run20)
  nn/              hitnet, chargenet, features, frame
  data/            RAT-DS ingest, the PMT-indexed memmap store, time-shuffle augmentation
  calibrate/       marginal-grid Z(theta) normalization
  receipts/        per-PMT charge, z-ring time, the model-dir runner, the (model,data) harness
  train/           resident / recipe / event / polish / moments / identities / nwj / reweight
```

**Every old import path still works**, once, with a `DeprecationWarning` naming its
replacement (`hitman/_compat.py`). The muonsync pipeline needed no edit. The move is
numerically inert; the full suite passing unchanged (200 passed, 9 skipped) is the receipt.

One real coupling was removed rather than shimmed: `hitman.npe.encoder` used to import the
WC feature scales, giving the library an edge into the application. Those two constants are
now local to the encoder and locked to the WC values by a test.

---

## The new lane: `hitman.ratio`

### Why it exists

The library had an NPE lane and no NRE lane. The measurable consequence: the muonsync NPE
arm is **491 lines** and its NRE arm is **~4,600**, because HITMAN owned the encoder, flow,
loss and SBC for one and `MLP.__call__` for the other.

The actionable gap was a single line. `hitman/train/loop.py`:

```python
marginal = jax.vmap(model)(obs, hyp[jax.random.permutation(key, hyp.shape[0])])
```

The marginal partner is hardcoded as a within-batch row permutation. That is correct exactly
when one row is one independent draw of θ. When rows are **grouped** — many observations
sharing one hypothesis, which is what simulation data looks like — two rows of the same group
can be paired with each other, leaving the group-level hypothesis untouched. The "negative"
is not a marginal draw, and nothing in the loss curve reveals it.

### The API

```python
from hitman.ratio import (
    RatioDataset, hash_split, concatenate, effective_sample_size, importance_weights,
    RatioModel, Standardizer, build_ratio_model,
    RowPermutation, GroupShift, GroupDerangement, build_group_index,
    FeatureSupport, compute_support, trust_radius, make_barrier,
    fit_ratio, RatioBundle, save_bundle, load_bundle,
)

ds     = RatioDataset.unweighted(x, theta, group_id, split, x_names=..., theta_names=...)
model  = build_ratio_model(ds, key=k)
result = fit_ratio(model, ds, key=k, pairing="group_shift")
sup    = compute_support(result.model, ds)
save_bundle("run/bundle", RatioBundle(model=result.model, support=sup))
```

Four design decisions worth knowing:

**1. `weight`, `log_q`, `source_id` are mandatory columns.** Not optional-with-a-default. An
unweighted single-source dataset is built with `.unweighted()`, which fills them explicitly.
You have to say it, so you cannot forget it. This is what makes multi-pool reweighting a
column operation later instead of an archaeology exercise.

**2. Negative pairing is a protocol** (`hitman.ratio.pairing`), with three implementations:
`RowPermutation` (the legacy behaviour, kept for WC), `GroupShift` (independent uniform
different-group partner per row; jit-traceable, the default) and `GroupDerangement` (shared
partner per source group; host-only, needs `np.unique`). The two group-aware ones guarantee
the negative never comes from the row's own group. They differ in the dependence structure
across rows of one group, so they do **not** agree seed-for-seed — validate a switch on
quality, never on bit-equality.

**3. Standardization lives in the model as frozen buffers**, not in a `cfg["stats"]` dict
that every consumer re-applies by hand. `stop_gradient` makes them non-trainable in effect
without a filter spec anyone can forget. The experiment owns the nonlinear feature
engineering (arcsinh, log, sin/cos); the model owns the affine standardization of it.

**4. A bundle is self-describing and refuses on mismatch.** Weights + specs + standardizers
+ support + provenance in one sha256-manifested directory, rebuilt from its own recorded
architecture. `assert_compatible` checks feature names **and order**, because every consumer
indexes positionally. This is the permanent fix for the bug class where a checkpoint and a
consumer disagreed about what the columns meant while both had the right width.

### `fit_resident` was evaluated for reuse and rejected

See `docs/RATIO_LANE_NOTES.md` RL-1 for the full reasoning. Short version: it has the right
seams for batching and weights, but its loss routes through the hardcoded pairing above, its
val split is row-level, and its contiguous-window batching would starve a group-aware
negative of distinct hypotheses. Those are contract changes, and making them in place would
move the calibrated WC receipts. `fit_resident` is untouched; the shared numeric core
(`nre_loss`) is genuinely shared.

---

## The new lane: `hitman.validate`

The validation chain as infrastructure rather than a notebook. Five gates over a
`(bundle, dataset)`, judged against a **declared** policy by the existing
`hitman.receipts.thresholds` engine — not a second verdict implementation:

| gate | what it answers | primitive |
|---|---|---|
| V1 calibration | is the logit a calibrated log-ratio at all | reliability / ECE |
| V2 self-normalization | `E_{p(x)p(θ)}[r] = 1`, exactly, for the optimal classifier | `self_normalization`, `fit_temperature` |
| V3 coverage | are the intervals honest | PIT of the truth under the ratio-induced 1-D posterior, + KS distance from uniform |
| V4 closure | inject a known value, recover it | pull statistics, robust and not |
| V5 OOD + ESS | how much of the operating point is outside training support; reweighting health | Mahalanobis occupancy per source, Kish ESS |

V3 is the one a BCE curve can never answer. A ratio estimator has no posterior *samples*, so
classical SBC ranks do not apply directly; the PIT form (CDF of the grid-normalized 1-D
posterior at the truth) is the continuous equivalent and needs no sampler.

`run_validation` emits a receipt whose top-level `status` is the answer. A gate that raises
is recorded as errored and does not abort the battery — nor can it turn a fail into a pass.

The default thresholds are **starting points chosen from what the metrics mean**, not from
measurements on real data. Expect to tighten them once a few runs establish the achievable
band. They are versioned with the code so a change of verdict is attributable to a commit.

---

## Proposals 1–5 (the previous contents of this file) are dead

They were an instinct pointed at the wrong layer: five injection seams added to the WC
receipt/flow/moment machinery, each verified bit-identical, and **each with exactly one
instantiation**. Generalization without a second consumer is speculation with tests.

| proposal | disposition |
|---|---|
| 1 — `HypSpec`/`ObsSpec` protocol | **kept**, and now genuinely used: the muon program builds its spec from `pool_schema`. But `MUON_HYP_SPEC`/`MUON_OBS_SPEC` shipped in `spec.py` are **stale** — they name `Xmu_max`/`width`, a parameterization the forward model retired on measurement. That is the drift class a spec is supposed to prevent, arriving via the spec. See RL-6. |
| 2 — pluggable density factors | **kept.** The one proposal that was real generalization: each factor owns its exact partition, and the monolithic `SplineMLE` is bit-reconstructible from the pieces to <1e-9. It has a downstream consumer waiting (`CountFactor` for the muon multiplicity term). |
| 3 — `ReceiptModel` / `strata_fn` harness | **dead as a story.** One model class, one `strata_fn`. The genuinely reusable bit was `percentile_strata`, which has been lifted into `hitman/receipts/strata.py` where the library can reach it; the rest moved to `hitman/wc/receipts/` with the observables it serves. |
| 4 — `ProjectionInstrument` | **dead.** One instrument (`RAY_INSTRUMENT`), WC-specific by construction. Moved to `hitman/wc/train/moments.py` unchanged. |
| 5 — encoder `feature_map` callable | **kept and used.** The one seam a second detector actually plugged into (`npe_arm.muon_feature_map`). |

What replaced them is this plan: split the application out, then build the two lanes the
library was actually missing — driven by a consumer that exists, not by a hypothetical one.

---

## Upgrading

- Old import paths keep working; update them when convenient. `hitman.inference` →
  `hitman.wc.inference`, `hitman.nn.HitNet` → `hitman.wc.nn.HitNet`, and so on. Do not mix
  old and new paths for the same submodule in one process (see RL-9).
- For new ratio-estimation work use `hitman.ratio`, not `hitman.train.fit`. The latter
  remains correct for ungrouped data and is what WC uses.
- Ship a `RatioBundle`, not a bare `.eqx` plus a JSON of statistics.
- Run `hitman.validate.run_validation` at the end of every fit and keep the receipt.
