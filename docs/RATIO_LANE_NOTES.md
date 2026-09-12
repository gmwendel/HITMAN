# Ratio-lane migration — execution notes

Same protocol as `muonsync/docs/V2_EXECUTION_NOTES.md`: what the plan said, what the code
actually turned out to be, what was deferred to the owner, and what was found and *not*
fixed. Written while executing, not after.

Branch: `feature/ratio-lane` (HITMAN), `master` (muonsync). Nothing pushed.

---

## RL-1 — `fit_resident` reuse: evaluated, rejected, with reasons

The plan asked to evaluate reusing `hitman.wc.train.resident.fit_resident` as the one
group-aware loop before writing a new one. Verdict: **reuse the ideas, not the function.**

What it has that is genuinely right, and was carried over: the `make_batch` seam, per-batch
weight threading through `_unpack_batch`, best-on-val checkpointing plus kept trajectory
snapshots, and device residency with the arrays passed as arguments rather than closed over.

What blocks reuse — the first is decisive:

1. **No pairing seam.** Its loss goes through `hitman.train.loop._batch_loss`, which builds
   the marginal as `hyp[jax.random.permutation(...)]`. That is the one line the whole
   downstream fork exists because of. There is no parameter to override it.
2. **Row-level validation split** (`resident.py:179-183` takes the last `val_fraction`
   rows). For WC that leaks one event; for grouped rows it leaks catastrophically.
3. **Contiguous row-window batches.** Coalesced reads are right for a row-independent
   dataset. When rows are sorted by group, a window spans only a handful of groups, so a
   group-aware negative would be drawn from a pool of a few distinct hypotheses — a
   materially worse marginal than a full permutation gives.
4. `DeviceData` is a WC container (`pmt_id` / `t` / `charge`).

Items 1–3 are contract changes, not parameters. Changing them in place would alter the
marginal construction and the val split **underneath the calibrated WC receipts and their
frozen `W`** — a physics change for that program, to serve a different one. So
`fit_resident` is untouched, and `hitman/ratio/train.py` is a second loop that shares the
genuinely shared part (`hitman.train.losses.nre_loss`). The verdict is recorded in that
module's docstring so the next reader does not re-litigate it.

**Consequence, recorded honestly:** there are now two training loops in the repo. That is
the intended end state only if WC eventually migrates onto the ratio lane; until then it is
duplication with a stated reason. Revisit if/when WC's receipts are ever recalibrated
anyway — at that moment the val-split fix (RL-2) becomes free.

## RL-2 — E-7 (group-unaware val split) was NOT applied to the WC loops

The plan said `fit_resident`'s val split "needs the E-7 fix either way". On inspection it
does not — not without a physics decision:

- WC events are written in i.i.d. order and hits are contiguous per event, so the last-rows
  split leaks exactly one event's worth of hits across the boundary. That is a real but
  tiny leak.
- Fixing it changes which rows are in validation, hence the val loss, hence **which epoch
  is selected as best**, hence the shipped WC model. The calibrated receipts and their
  frozen `W` are downstream of that choice.

So: `hitman/ratio/train.py` is group-aware from birth, and `fit_resident` keeps its
behaviour. **DEFERRED-TO-USER:** should WC adopt a group-aware (per-event) validation split
at its next recalibration? Cost is one recalibration; benefit is removing a known
one-event leak.

## RL-3 — E-1 is not a bug (correction absorbed)

Recorded here because the review that preceded this work called it one. `fit_pairs` running
with `support=None` is a **ratified decision**: the barrier is anchored per pair at
`d0 = max(train q99, d_M(pair MAP) + 2)` precisely because 6 of 155 operational MAPs are
themselves out of distribution, and a support-guarded ensemble fit would silently re-fit
exactly those six — changing a published number rather than protecting it. The corner study
carries the guard; the six are reported as `ood_fit`.

The same anchoring rule is now encoded in the library as
`hitman.ratio.support.trust_radius(support, anchor_feat, margin=2.0)`, so the reason
travels with the code rather than living only in a campaign document.

**DEFERRED-TO-USER:** enable the guard in `fit_pairs`? Argument for: `_make_objective`'s own
docstring documents a spurious `delta_t = +1290 ns` global minimum on pair 151 that beats
the physical solution by 14.5 in objective while sitting 3.5 z-units from its nearest
training neighbour, and `make_multistart` draws starts over ±300 ns, so the fit does search
there. `dt_spread`/`multimodal_mask` catch a pair whose starts disagree, but a pair where
the spurious basin wins from *every* start is recorded as a clean, confident, wrong δt.
Argument against: it is a physics change and would move published numbers. If taken, both
variants must be reported side by side. A loud comment at the call site now states all of
this; behaviour is unchanged.

## RL-4 — E-6 recorded, not fixed (physics-affecting)

`Σ_i log r(x_i | Θ)` is a proper log-likelihood-ratio only if muons are i.i.d. given Θ.
Two known violations: the station's **multiplicity** is Θ-informative and is discarded
entirely, and muons sharing a `lineage_idx` (same depth-1 branch — a materialized v3 column)
are correlated, so the sum over-counts independent information and the Laplace `sigma_dt`
from `inv(H)` is **anti-conservative**. Independently corroborated: a pair-72 profile
measured `σ_Laplace = 2.09 ns` against a flat ±8 ns profile.

Not fixed — changing the objective changes every published number. Recorded as the ratio
lane's **first post-migration science task**. The machinery is already in place:
`hitman.density.CountFactor` (NB2 with an E-dependent dispersion and a monotone yield head,
tested) supplies the count term, and `RatioModel`'s reserved `encoder` slot is where a
set-valued scorer would go. `log L = count.log_prob(N|Θ) + Σ_i log r_i` is strictly more
information than the current objective and needs no encoder change — that is the cheap half,
and it should be attempted first, behind the V1–V5 gates.

## RL-5 — the plan's archaeology list was partly wrong; two deletions refused

The plan said to delete "phase3b loader, legacy split, consolidated-pool adapter paths, the
dead helper". Three of four were done. The fourth was refused after checking consumers:

| item | status | evidence |
|---|---|---|
| `nre_arm._build_shower_to_rows` | **deleted** | its own docstring said `train_nre` no longer calls it; only a test used it |
| `nre_arm._shower_from_phase3b_dir` + the phase3b branch | **deleted** | predates the vertex writer, can never supply h1, which Θ now requires unconditionally |
| `--split legacy` (5 branches) | **deleted** | kept a bucketing with station geometry inside `x` alive in the live training path |
| `hitman_adapter._read_run_dir` (Gen-1) | **KEPT** | live consumers: `twostation.py:174`, `fisher_forecast.py:564` — the DENSITY arm reads that format |
| `hitman_adapter.from_consolidated` / `_load_consolidated_showers` (Gen-2) | **KEPT** | live consumer: `train_surrogate.py:184`, plus `test_consolidate_pool.py` |

The last two are Gen-1/Gen-2 *by generation* but they are the density arm's live readers,
not the NRE arm's. Deleting them would have broken a working arm to tidy a different one.
They should be retired when the density arm migrates, not before.

## RL-6 — `hitman.spec.MUON_HYP_SPEC` is stale and is now unused

It names `(cos_zen, Xmu_max, width, X1, r, psi)`. `Xmu_max`/`width` are the Gaisser–Hillas
peak and λ that the forward model **retired on measurement** (λ was not identified: it ranged
1–18 g/cm² at equal χ² while X0 ran to −1e5). Pool v3 emits `Xmu_prod_mean`/`Xmu_prod_sd`
instead.

Rather than edit it — `hitman_adapter` imports it, and it is locked by `tests/test_spec.py` —
`ratio_data.muon_hyp_spec()` **builds** the spec from `pool_schema`, and a test
(`test_hitman_muon_spec_is_stale_and_we_do_not_use_it`) pins the divergence so nobody
imports the stale one by reflex. **DEFERRED-TO-USER:** delete `MUON_*_SPEC` from
`hitman/spec.py` outright once `hitman_adapter` is migrated? It is a genuine trap: it is the
library's *advertised* muon instantiation and it is wrong.

## RL-7 — three Θ layouts existed; there are now two, and only one is authoritative

Before: `hitman.spec.MUON_HYP_SPEC` (stale), `hitman_adapter.MUON_ENERGY_HYP_SPEC`
(`cos_zen, log10_E, X1, r, psi` — the density arm's), and `nre_arm`'s `cfg` names (the one
actually trained). Now the ratio arm derives its Θ from `pool_schema`, so the NRE side has
exactly one authoritative layout. `MUON_ENERGY_HYP_SPEC` remains for the density arm.
The trained NRE Θ **changed** in the process: X1 was added, so it is 7-dimensional
(`log10_E, cos_zen, X1, log_h1, log_r, sin_psi, cos_psi`) against the old arm's 6. Any
comparison against an nre7-era checkpoint is a rough reference, not a baseline — which
decision 1 of `FORWARD_MODEL_V2_PLAN.md` already accepted.

## RL-8 — two bugs found *in the new code* while testing it

Recorded because both are the kind that pass a smoke test:

1. `jax.random.fold_in` rejects a negative seed (`OverflowError: -20 out of bounds for
   uint32`). The validation stream was folding in `-step` to keep it distinct from the
   training stream. Fixed with a separate key.
2. **Clip-then-renormalize does not bound the weights.** `importance_weights` normalized to
   mean 1, clipped at `w_max`, then renormalized — and the renormalization scales the
   clipped values *back above the cap*. Measured `max = 10.66` for `w_max = 10.0`. The
   function reported a bound it did not hold to. Now iterated to a fixed point. The same
   pattern exists in `hitman/wc/train/reweight.py:44-51` (normalize → clip → renormalize);
   there it is applied to a histogram-derived weight where the overshoot is small, and it
   was left alone rather than perturbing calibrated WC weights. **Recorded, not fixed.**
3. **Validation was not chunked.** `run_val` sent the entire validation subsample (default
   cap 2M rows) through one jitted call, allocating a `[rows, width]` activation per layer
   *per class*. Invisible on test-sized data; an OOM on the first real run, and precisely
   the failure the arm being replaced had already chunked at 1M rows to avoid. Now chunked
   at `val_chunk_rows` (default 1M), with the negatives still drawn once for the whole
   subsample so the group-aware pairing sees the full group structure. A test pins that a
   deliberately ragged chunk size reproduces the one-shot metrics exactly.
4. **`compute_support` materialized the whole split on the device.** It pushed every row
   of `x` and `theta` through `jnp.asarray` purely to apply an affine standardization, then
   held the result in float64 — ~9 GB of device memory at pool scale to subtract a mean.
   Now streamed in numpy chunks with the standardizer buffers pulled to host; only
   `max_rows` of standardized data is ever resident. Extrema stay exact across chunk
   boundaries and the subsample is chunk-size-independent, both pinned by tests — otherwise
   the recorded support would depend on a performance knob.

Items 3 and 4 are the same failure class, and both were in code I wrote fresh while
replacing an implementation that had already solved them. Worth stating plainly: the
arm being replaced had learned these lessons from measurement, and a rewrite silently
discards that unless someone goes looking. Neither would have shown up in any test at
fixture scale.

## RL-9 — the compat shim's one accepted limitation

`hitman/_compat.py` rebinds `sys.modules[old] = new_module`, so an old path is the *same
object*, not a copy. For a shimmed **package**, though, a later
`import hitman.inference.mle` resolves through the aliased package's `__path__` and creates
a *second* module object for `hitman.wc.inference.mle`. Mixing old and new paths for the
same submodule in one process is therefore unsupported.

Nothing does: the muonsync pipeline imports none of the moved modules, and this repo's tests
and scripts were rewritten to the new paths. Per-submodule shims for the whole WC surface
would be a large permanently-maintained file set for a case that does not arise. Documented
in the module.

## RL-11 — the layering invariant was violated twice, invisibly

Both found by asserting the invariant, not by anything failing:

1. `hitman/npe/encoder.py` imported the WC feature scales for its default feature map.
   Fixed by making the two constants local (locked to the WC values by a test).
2. `hitman/npe/batch.py` imported `hitman.wc.train.identities`, so `import hitman.npe`
   pulled in eight application modules. Both `npe/batch.py` and `npe/train.py` are WC by
   construction (`data.pmt_pos[data.pmt_id[rows]]`, `theta[:, 5]`, `HitStore.hit_offsets`)
   and moved to `hitman/wc/npe/`. The detector-agnostic half of the lane — encoder, flow,
   loss, sbc, which is the half muonsync's `npe_arm` uses — stayed.

`tests/test_layering.py` now enforces it three ways: a **subprocess** import check (so a
module another test already imported cannot mask the edge), a **static AST scan** (so a
lazily-imported edge inside a function is caught too), and a check that the deprecation
shims are **aliases rather than copies** (a copy would silently break `isinstance` across
the two paths). Without this the boundary erodes one convenient import at a time, and it
had already started to.

## RL-10 — what is NOT yet migrated

Stated plainly so nobody assumes otherwise:

- **`nre_arm.py` still has its own training loop.** `ratio_data.py` produces a
  `RatioDataset` and the lane trains on it (proven by the end-to-end smoke), but
  `nre_arm.cmd_train` has not been rewired to `fit_ratio`. Doing so changes the trained
  model (different pairing semantics on the device path, different Θ dimension), so it is a
  physics-affecting step that deserves its own before/after comparison rather than being
  folded into a mechanical migration.
- **`twostation_nre` / `twostation_corners` still reconstruct from `summary.json`.** The
  E-2 hotfix (coords + feature-name/order validation) closes the silent-mismatch hole; the
  structural fix is to load a bundle. That migration also wants the two-station hypothesis
  extended to carry X1 as a nuisance (8 → 9 parameters), which is physics-affecting.
- **`hitman_adapter.py` is untouched** beyond what the split required.

The seam is proven and the lane is tested; the consumers' migration is the next commit, and
each remaining step is one that changes numbers.

## Validation status at the end of this work

| suite | result |
|---|---|
| HITMAN full suite after the split, before the new lanes | 200 passed, 9 skipped, 0 failed |
| HITMAN full suite, final (all fixes in) | **270 passed, 9 skipped, 0 failed** (28m17s) |
| HITMAN new tests (`test_ratio_pairing` 15, `test_ratio_lane` 33, `test_validate` 19, `test_layering` 3) | 70 |
| muonsync `test_ratio_data`, `test_nre_arm`, `test_twostation_nre`, `test_npe_arm` | 80 passed |
| muonsync `test_twostation_corners`, `test_hitman_adapter`, `test_hygiene`, `test_consolidate_pool` | 61 passed |
| muonsync `test_receipts_battery` | 10 passed |
| all 16 muonsync pipeline modules import with ZERO deprecated-path use | verified |

The full muonsync suite has not been run end to end — a 16-core campaign is using the box,
and the remaining modules (`test_orchestrate`, `test_extract`, `test_d2_map`, …) are
untouched by this work. Run it when the cores free up.

---

# Session 2 — the lane on real data, and what it found

Scope: exercise the ratio lane end to end on the 114.7M-muon production cache, compare
against the trained `nre7` baseline, run the validation chain for the first time, and
attack the two-station MLE fragility.

## RL-12 — the headline: `nre7`'s two-station results were computed on the wrong observables

The E-2 hotfix from session 1 fired on real data, and it was not a false alarm.

`nre7` was trained with `--coords pivot`, so its observable block is
`(u_w, log_h_ang, alpha_t)`. `twostation_nre._muon_features_jax` implemented only the raw
block `(dt_w, alpha_r, alpha_t)`. Both are width 3; `derive_nre_static_config` computes the
raw statistics unconditionally, so every `stats[...]` lookup succeeded; the total feature
width is 9 either way. **Nothing raised.** Every stored `twostation_nre7` number — 200
pairs, every `sigma_dt`, the pull robust-SD of 0.692, pair 72's 2.09 ns, the OOD-MAP
count — was produced by scoring the model on three observables it had never seen.

Magnitude, measured on 200k real validation rows: decorrelating only the two pivot
observable columns moves the logit from mean **+4.94 (sd 1.86)** to mean **−761 (sd 1248)**,
correlation **0.08**. The observable block carries essentially all the ratio signal.

`pivot_coords.to_pivot_jax` already existed and its docstring anticipates exactly this
consumer; it had simply never been wired in. Now implemented and locked against
`nre_arm.featurize` — the lock the raw path always had and the pivot path never did.

Re-running the identical 200 pairs (seed 0; `dt_star` reproduced exactly, asserted):

| | stored (buggy) | corrected |
|---|---|---|
| pull robust-SD | 0.692 | **0.912** |
| resid robust-SD | 32.85 ns | **25.67 ns** |
| median σ(δt) | 41.16 ns | **26.58 ns** |
| ill-posed | 19.5 % | 23.0 % |
| multimodal | 71.5 % | 84.5 % |
| OOD MAPs (well-posed) | 6/155 (3.9 %) | 37/154 (24.0 %) |

**Most of the pull deficit was this bug.** DEFERRED-TO-USER: what to republish. Note the
corrected fit lands out-of-distribution far more often (24 % vs 4 %), so the support guard
matters *more* now, not less — which strengthens rather than weakens RL-3's open question.

## RL-13 — T4 is ill-posed as specified, twice over, and the measurements say so

Asked for: a cluster-robust sandwich clustered by shower, optionally by `lineage_idx`.

1. **`lineage_idx` does not exist.** The pool is `muonsync-pool-v2`; `lineage_idx` is a v3
   column. Unreachable, full stop.
2. **One pair is one shower**, so clustering by shower gives exactly **one cluster per
   fit** — the meat `(Σs)(Σs)ᵀ` is rank 1 and, at a MAP where the score vanishes, carries
   no information. Not estimable.
3. **97 % of well-posed pairs have exactly 2 muons** (191/200 are 1+1; the "155" in the
   brief is precisely the 1+1 well-posed subset). The meat `Σᵢ sᵢsᵢᵀ` is a sum of `n_mu`
   rank-1 terms in an **8-dimensional** parameter space, so `rank(B) ≤ 2 < 8`. A per-pair
   sandwich cannot estimate an 8×8 covariance from 2 observations.

What *is* well-posed — an exchangeable-correlation meat with ρ supplied externally rather
than estimated, `B(ρ) = (1−ρ)Σᵢsᵢsᵢᵀ + ρ(Σᵢsᵢ)(Σᵢsᵢ)ᵀ` — was computed anyway, on the
**corrected** fit, at ρ ∈ {0, 0.55, 0.59}:

| estimator | median σ(δt) | pull median | pull robust-SD | >2× vs Laplace |
|---|---|---|---|---|
| Laplace `A⁻¹` | 26.58 | −0.067 | **0.918** | — |
| Huber-White (ρ=0) | 14.76 | −0.099 | **1.978** | 17 |
| ρ = 0.55 | 16.01 | −0.086 | **1.890** | 18 |
| ρ = 0.59 | 15.57 | −0.086 | 1.899 | 18 |

**The prediction failed, in an informative way.** The sandwich does not widen σ(δt) — it
*halves* it (median ratio 0.506, IQR [0.23, 1.08]) and the pull **overshoots to ≈1.98**.
That is the signature of a rank-deficient meat, not a physical correction.

**The ρ effect in isolation is +11 %** (ρ=0.55 vs ρ=0, median ratio 1.106), not the
predicted `1/(1−ρ) = 2.22×`. This *confirms* the sum-vs-difference reasoning by measurement:
δt shifts only station B, a shower-common mode shifts both stations together and cancels in
the A−B difference, so δt is difference-coupled and nearly blind to the common mode.

And the direction was never right: the pull was **0.912 < 1**, i.e. σ already slightly
*over*-estimated. Inflating it moves *away* from 1.

**Recommendation (DEFERRED-TO-USER):** do not adopt a per-pair sandwich for `fit_pairs`. It
is not estimable at N=2 and the numbers confirm it. `fit_pairs`' published outputs are
unchanged. The correlation problem is real but it lives where many muons are pooled — see
RL-14 — not in a 1+1 two-station δt.

## RL-14 — the validation chain, run on a real model for the first time

`hitman.validate` on the bundle trained through the bridge (val split; the v1 prep bundle
never stored its test rows, so the gates share the split early stopping used — a real
limitation of the bridge, not of the chain):

| gate | value | verdict |
|---|---|---|
| calibration.ece | 0.00194 | **pass** |
| calibration.temperature | 0.961 | **pass** |
| calibration.self_norm | **0.679** | **FAIL** (target 1) |
| coverage.ks_uniform | **0.961** | **FAIL** |
| coverage.max_coverage_deviation | **0.919** | **FAIL** |
| ood.frac_out_of_distribution | 0.000985 | pass |
| ess.min_source_ess | 1.0 | pass |

The classifier is beautifully calibrated *as a classifier* (ECE 0.0019) and badly
mis-normalized *as a ratio* (E[r] = 0.68, not 1). Coverage is catastrophic: empirical
coverage is 0–3 % at every nominal level.

**This is not a broken gate — it is measuring the correlation pathology, from a completely
independent direction.** Coverage of the ratio-induced posterior as a function of how many
muons of a shower are pooled:

| muons/shower | 1 | 4 | 16 | 64 | 256 | 7241 |
|---|---|---|---|---|---|---|
| KS from uniform | 0.508 | 0.872 | 0.937 | 0.987 | 0.988 | 0.988 |
| frac PIT at 0/1 | 0.15 | 0.75 | 0.95 | 1.00 | 1.00 | 1.00 |

Monotone collapse with pooling — exactly what treating correlated observations as
independent predicts, and the same physics as the ICC = 0.625 measurement, reached without
any residual model. It corroborates E-6/RL-4 quantitatively.

Note also that even at **one** muon per shower the PIT is non-uniform (mean 0.178), so
there is a per-muon bias *on top of* the over-counting. Two separate defects.

## RL-15 — the ratio lane reproduces the baseline (RL-10's evidence)

Same data, architecture (128×3 mish, 34,433 params), batch (262,144), lr, clipping:

| | nre7 @ 6118 | ratio lane @ 6000 |
|---|---|---|
| val BCE (nre_arm units) | 0.14139 | **0.14043** |
| val AUC | 0.99615 | **0.99625** |

**A unit trap worth recording:** `hitman.train.losses.nre_loss` carries a factor ½ that
`nre_arm.nre_bce_loss` does not. Compared raw, the new loop looks twice as good. Always
convert before comparing.

Step time and memory: see the commit message on `4e2e417`. Summary — host **730 → 352
ms/step** (new is ~2× faster; 298 with the old loop's own pairing), device **NEW/OLD ≈
1.35×** (slower, under the 1.5× bar), peak memory equal to within 3 %. A micro-benchmark
shows the loss+grad+update step is *identical* (8.64 ms) across old model, new model and
new-model-with-old-loss, so the device gap is dispatch overhead, not the model or the four
extra standardizer leaves.

## RL-16 — this box's GPU cannot allocate above ~3.6 GB

`nvidia-smi` reports 28 GB free with no compute processes; JAX reports a 25.6 GB limit; raw
FP32 GEMM runs at a healthy **69.5 TFLOP/s** at 462 W. But every allocator (BFC,
`cuda_malloc_async`, `platform`) tops out near **3.6 GB**, and the 0.60 preallocation fails
outright. Compute is fine; large allocation is not. WSL2 host-shared memory is the likely
cause.

Consequences, all honest rather than worked around: preallocation was disabled (which keeps
the 0.60 fraction as a *cap* and uses strictly less than the policy allows); device-resident
training on the full 114.7M rows is impossible, so the device comparison used a
group-aligned 20M-row subsample and the production training ran host-resident; and absolute
step times are **not** comparable to the recorded 2.45 ms/step baseline. The old loop
re-measured under identical conditions is the valid reference.

## RL-17 — my own v3-only changes made the v2 production pool unreadable

Session 1 retired the Gen-1 loader and required v3 events columns; R0 bumped the
shower-cache schema for `particles.weight`. Together those make
`muonsync_pool_1k` — a `muonsync-pool-v2` pool, and the only pool with a trained model and
stored two-station results — unopenable by `nre_arm`: the 14 GB shower cache misses on the
new schema, and the fallback needs `Xmu_prod_mean`/`h1_m`, which v2 does not have.

Deliberate at the time, but it is a real capability regression against existing data, and
it is why this session reads that cache through a one-off bridge in scratch
(`v2_cache.py`) rather than the production path. **DEFERRED-TO-USER:** either accept that
v2 pools are read-only history, or add an explicit narrow v2 reader. Do not let it happen
silently a third time.

## Validation status

| suite | result |
|---|---|
| HITMAN full suite | 270 passed, 9 skipped, 0 failed (pre-session-2 baseline) |
| muonsync `test_twostation_nre`, `test_ratio_data`, `test_nre_arm` | 80 passed |

## RL-18 — T5: most of the 0.625 ICC was residualizer lack-of-fit, but the design effect is still enormous

Stage A measured intra-shower ICC = 0.625 on the `u_w` residual after a cross-fitted
quadratic in Θ. Redone with the **trained model's own per-muon score** `∂ log r/∂θ` at each
shower's true Θ (autodiff, 300 val showers × 400 muons) — the quantity that actually
inflates the variance of the summed score:

| Θ direction | ICC | design effect (n̄=400) | n_eff/N |
|---|---|---|---|
| log10_E | 0.095 | 39 | 2.6 % |
| cos_zen | 0.075 | 31 | 3.3 % |
| log_r | 0.098 | 40 | 2.5 % |
| sin_psi | 0.185 | 75 | 1.3 % |
| cos_psi | 0.211 | 85 | 1.2 % |
| h1 | 0.073 | 30 | 3.3 % |

**Both halves of the answer matter.** The ICC drops from 0.625 to 0.07–0.21, so *most* of
the Stage A number was the quadratic residualizer failing to condition on Θ, not genuine
Θ-insufficiency — the trained model conditions far better. But ICC does not have to be
large to be fatal when you pool: at ~400 muons the design effect is **30–85×**, so the
effective sample size is 1–3 % of nominal and the factorized posterior is overconfident in
σ by √30–√85 ≈ **5.5–9.2×**.

That independently and quantitatively explains RL-14's coverage collapse: with σ
understated 5–9×, the truth essentially never lands inside the interval, so the PIT pins at
0 and 1 — which is exactly what was observed.

The two highest-ICC directions are `sin_psi`/`cos_psi`, i.e. the azimuthal structure is the
most shower-common component. Physically sensible: a shower's azimuthal asymmetry is a
whole-shower property, not a per-muon one.

**Consistency check across three independent routes** — residual correlation (Stage A),
posterior coverage vs pooling (RL-14), and score ICC (here) — all say the same thing: the
per-muon factorized likelihood over-counts independent information by 1–2 orders of
magnitude once a whole shower is pooled. It is *not* what limits the 1+1 two-station δt
(RL-13), because there only two muons are pooled and δt is difference-coupled.

---

# Session 3 — is delta_t identified by the likelihood, or by the priors?

CPU only, per instruction. Study only: `PRIOR_SD` and `fit_pairs` defaults are untouched;
`PRIOR_SD` is a module global, patched at runtime inside the analysis script and always
restored. Results in `data/runs/prior_identifiability/` (gitignored), scripts in scratch.

**Answer: overwhelmingly the priors.** Details below.

## RL-19 — design choice that makes the scan mean anything

`make_multistart` draws its start dispersion **from `PRIOR_SD`**, so naively scaling the
prior widens the STARTS too and confounds "delta_t is prior-pinned" with "the optimizer
lost the basin". Starts are therefore generated ONCE at nominal width and held fixed at
every scale; only the objective's prior varies. The scaling-starts variant is reported
separately as the failure-mode check, and it does exactly what you would fear (below), which
is why the isolation mattered.

## RL-20 — T1: the prior-width scan (the headline)

200 pairs, corrected pivot featurization, `PRIOR_SD[0:7] *= s`, `PRIOR_SD[7]` fixed at 1e5.

| s | median σ(δt) ns | resid robust-SD ns | pull robust-SD | well-posed | multimodal | OOD MAPs |
|---|---|---|---|---|---|---|
| 1 | **27.7** | 25.6 | **1.046** | 155 | 123 | 38 |
| 3 | 96.1 | 46.1 | 0.578 | 131 | 52 | 122 |
| 10 | 165.8 | 69.6 | 0.406 | 120 | 32 | 117 |
| 30 | 133.1 | 63.7 | 0.375 | 130 | 40 | 130 |
| 100 | **161.9** | 59.7 | 0.440 | 123 | 33 | 122 |

**Intermediate case, and the saturation scale is s ≈ 10.** σ(δt) rises steeply to ~150 ns
and then stops responding: 10×, 30× and 100× wider priors all give the same answer within
scatter. So the likelihood *does* eventually identify δt — at **~150 ns**, not 27.7 ns.

In information terms `(150/27.7)² ≈ 29`, so **roughly 97 % of the quoted Fisher information
on δt comes from the nuisance priors** and ~3 % from the marks.

Two secondary observations, both worth keeping:
* the pull is **1.046 at nominal** — the reported uncertainty is honest *conditional on the
  priors being right* — and falls to ~0.4 once they are loosened, i.e. σ inflates faster
  than the residual does;
* the residual robust-SD itself grows 25.6 → ~60 ns, so the tight priors are genuinely
  improving the point estimate, not merely shrinking the error bar.

**Answer to "how well must we know the geometry":** the current priors are ~10× tighter than
the width beyond which they stop mattering. Loosen them by more than ~3× and the measurement
degrades immediately.

## RL-21 — T2: δt lives in the likelihood's null space

Likelihood-only Hessian (no prior) at each corrected MAP, 200/200 usable:

* near-null directions per pair (|λ| < 1e-3 λmax): **median 7 of 8** (156 pairs have 7,
  34 have 6). Negative-curvature directions: median 0, present in 21 pairs.
* median |λ| spectrum normalized to λmax:
  `1.00e0, 2.06e-4, 4.81e-7, 7.35e-8, 1.52e-9, 2.75e-10, 2.00e-11, 8.30e-13`
  — the likelihood is effectively **rank 1**.
* **overlap of `e_δt` with the near-null subspace: 1.0000 at every percentile from p10 to
  p90**; 99 % of pairs above 0.9, none below 0.1.

So the prediction is confirmed and then some: δt is not merely partly in the null space, it
is essentially entirely in it.

**A number NOT to quote.** The ridge-regularized likelihood-only σ came out 0.47 ns
(ridge 1e-6·λmax), 0.05 ns (1e-4) and 0.00 ns (1e-2). These are artifacts: they scale
exactly as `1/√ε` (a 100× ridge gives a 10× smaller σ — 0.47 → 0.05 — which is what the
numbers show), so they measure the ridge, not the data. A ridge on a null direction *is* a
prior. The meaningful likelihood-only figure is T1's saturation value, **~150 ns**.

## RL-22 — T3: it is NOT the core/clock degeneracy

Scaling **only** `PRIOR_SD[3:5]` (core x, y):

| s | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|
| median σ(δt) ns | 27.7 | 33.7 | 29.6 | 28.6 | 29.2 |

Flat. A 100× wider core prior does nothing. The core/clock degeneracy the literature
identifies is **not the operative one at this geometry**, and the reason is the same
difference-coupling that defeated the correlation story in RL-13: the two stations are ~100 m
apart, so a core shift moves both arrival times almost equally and cancels in the A−B
difference that δt measures.

**So which prior is doing the work?** Widening each nuisance alone by 30×:

| nuisance | σ inflation |
|---|---|
| `d_log10E` | **3.53×** |
| `d_log_h1` | **1.81×** |
| `d_core_y` | 1.14× |
| `d_phi`, `d_T0`, `d_cos_zen`, `d_core_x` | 0.94–0.99× |
| all seven | 4.80× |

δt is pinned by **assumed knowledge of the primary energy and the first-interaction
height** — the two quantities that set the expected arrival-time distribution — and by
nothing else. Physically sensible, and actionable: it says the deployment question is
"how well can we know E and h1 per shower", not "how well can we survey the core".

## RL-23 — failure mode when the starts widen too

With starts scaled alongside the prior (the naive scan):

| s | median σ ns | resid rSD ns | pull rSD | multimodal |
|---|---|---|---|---|
| 10 | 585 | 69 | 0.140 | 15 |
| 100 | 5958 | 161 | 0.021 | 3 |

`dt_hat` does not run away to infinity (the fits still return), but σ explodes and the
multimodality flag collapses — the multistart no longer brackets the basin, so the spread
diagnostic stops firing precisely when it is most needed. Reported rather than tuned around.

## RL-24 — the caveat sentence for any writeup

> σ(δt) ≈ 27 ns is a **posterior width conditional on the assumed nuisance priors**, not a
> likelihood-only measurement. For a 1+1-muon pair the per-muon factorized likelihood
> constrains essentially one direction of the 8-parameter space, and δt is not in it — the
> overlap of the δt direction with the likelihood's near-null subspace is 1.00, and 7 of 8
> directions are near-null. Widening the nuisance priors inflates σ(δt) to ~150 ns, where it
> saturates, so roughly 97 % of the quoted information on δt comes from the priors. That
> constraint is dominantly the assumption that the primary energy (3.5×) and
> first-interaction height (1.8×) are known to their nominal widths; the core position
> contributes nothing measurable. The number therefore describes how well we assume the
> shower is known, not what two 1 m² stations measure.

**DEFERRED-TO-USER**, all physics-affecting, none applied:
1. whether to quote σ(δt) at all without this caveat attached;
2. whether to report the prior-free ~150 ns alongside it as the honest floor;
3. whether the two-station programme should be re-scoped around per-shower E and h1
   knowledge — which is what actually buys the precision — rather than around core survey.

## RL-25 — are the nuisance priors ORACLES? (user question: can we assume a prior at all?)

The priors are centered on truth: `log10_e = log10_E_true + d`, `core_x = d_core_x`,
`h1 = h1_true·exp(d)`, all with `d = 0` meaning "exactly right". That is an oracle, not a
prior — at deployment you have an *estimate* with its own error, not the truth.

Tested directly: same widths, but centered on truth + a draw from the prior, i.e. an
unbiased external estimate whose resolution equals the prior width (the best case you could
actually field). 200 pairs, three independent draws.

| centering | median σ(δt) | resid robust-SD | resid median | pull robust-SD |
|---|---|---|---|---|
| oracle (as shipped) | 27.7 | 25.6 | **−2.61** | 1.046 |
| noisy, resolution = prior (rep 1) | 43.1 | 28.2 | −0.85 | 0.724 |
| noisy, resolution = prior (rep 2) | 35.4 | 26.9 | −0.73 | 0.646 |
| noisy, resolution = prior (rep 3) | 37.8 | 27.7 | +1.31 | 0.660 |
| noisy, external 3× tighter than prior | 22.1 | 19.8 | −0.22 | 0.833 |

**My prediction was wrong, and the direction is reassuring.** I expected the residual to
grow and the pull to blow past 1 (σ exposed as dishonest). Instead: σ inflates only ~35 %
(27.7 → ~38), **the residual is essentially unchanged** (25.6 → 26.9–28.2), and the pull
falls to ~0.65 — the reported uncertainty becomes *conservative*, not optimistic.

So oracle-centering is **not** the dominant defect. Mis-centering a nuisance by one prior
width costs little because the fit re-optimizes; what costs is letting the nuisance wander
many widths, which is what RL-20's width scan varied. **The width is the load-bearing
assumption, not the center.** That is a much better position to be in: a width can be
defended from an independent instrument's resolution, whereas a center on truth cannot be
defended at all.

Two riders:
* the oracle run has `resid_median = −2.61 ns` and the noisy runs are closer to zero, so
  the truth-centering appears to *introduce* a small bias rather than remove one;
* that bias matters more than σ for any stacked measurement — see RL-26.

## RL-26 — what this implies for tracking a real, unknown drift

δt's own prior is flat (1e5 ns vs a ~27 ns answer), so nothing here assumes knowledge of the
drift; the study injects a known `dt_star` and measures recovery. For deployment the
relevant quantity is not the per-pair σ but the stacked one: a clock offset common to N
pairs scales as σ/√N, so even the prior-free ~150 ns floor (RL-20) gives ≈ 150/√155 ≈ **12 ns**
over the existing 155-pair sample, with no tight per-shower priors at all.

**The binding constraint on a stacked measurement is therefore bias, not variance.** A
residual median of a few ns that does not average away sets a floor no amount of stacking
clears, and RL-25 measures exactly such an offset (−2.6 ns oracle-centered). Quantifying and
removing that offset is worth more than tightening any nuisance prior.

**DEFERRED-TO-USER, and the question that decides legitimacy:** where would per-shower E and
h1 knowledge come from? If from an independent detector (a surface array), the prior is
legitimate and its width should be that instrument's real resolution. If from the same
muons, it is circular and the 27 ns is double-counting. This study cannot settle that — it
is an experiment-design question — but it is now the question that matters.

## RL-27 — TWO DETECTORS ONLY: the honest number is 44 ns, not 150

Constraint from the user: assume only the two stations exist, no external per-shower
information. The question "can we assume a prior?" then has a specific answer — **yes, the
population prior**. With no external instrument you still know the cosmic-ray flux, the
zenith distribution and the atmosphere, so the defensible prior is the shower POPULATION
distribution centred on the population MEAN. That is knowledge about the ensemble, not
about this shower, and it requires no second detector.

Population moments read off the NRE's own standardization statistics (they *are* the
training-population moments): log10E sd 0.409, cos_zen sd 0.115, log h1 sd 0.341, median
core distance 234 m. Compared with the shipped `PRIOR_SD`, the shipped widths are already
near-population for the two parameters that matter (E 0.82×, h1 0.68×) and are too tight
only for T0 (10×) and core (2.3×) — which RL-22 showed contribute nothing.

Three configurations, separating width from centring:

| config | median σ(δt) | resid robust-SD | resid median | pull robust-SD | n | σ/√n |
|---|---|---|---|---|---|---|
| A shipped (truth-centred oracle) | 27.7 | 25.6 | −2.61 | 1.046 | 155 | 2.23 |
| B population widths, truth-centred | 30.8 | 22.1 | −0.29 | 0.790 | 153 | 2.49 |
| C **two-detector** (pop width + pop centre) | **44.2** | **27.5** | −1.02 | 0.790 | 146 | **3.66** |

**Removing the oracle entirely costs a factor 1.6 in σ, and essentially nothing in the
actual residual** (25.6 → 27.5 ns). The pull is 0.79, i.e. conservative.

This *corrects the framing* of RL-20. The "~97 % of the information comes from the priors"
statement is true against a **no-prior** limit, but that limit is not the two-detector
scenario — it is absurd (s=100 means a log10E prior of 50, i.e. energies spanning 10⁵⁰).
The relevant comparison is against the **population** prior, and there the answer is 44 ns.
Both numbers are correct; the population one is the one to quote.

Two riders, one of which makes 44 ns an **upper bound**:

* The pair sample is selection-biased toward high energy (more muons → more likely to form
  a pair): the true log10E of the selected pairs sits **+0.587** above the population mean,
  1.4 population-σ. Config C hands the fit that miscentred prior anyway, so it is penalised
  by the selection. A prior conditioned on the trigger would centre higher and do better.
* Stacking a common offset over the 146 pairs gives σ/√n ≈ 3.7 ns (or resid/√n ≈ 2.3 ns),
  **but the residual median is −1.02 ns**, and a bias that does not average away is a floor
  no stacking clears. For a two-detector drift measurement the bias, not the variance, is
  what needs work.

**Revised caveat sentence, superseding RL-24 for the two-detector case:**

> With two stations and no external per-shower information, σ(δt) ≈ 44 ns per 1+1-muon
> pair, using only the known shower-population distribution as the prior. The 27 ns figure
> assumes each shower's energy and first-interaction height are known a priori to ~0.5 and
> is an oracle, not an achievable measurement. Neither number is limited by knowledge of the
> core position. Stacking a drift common to ~150 pairs reaches the few-ns level, at which
> point a ~1 ns residual bias, not σ, is the binding constraint.

---

## RL-28 — verifying the ratio estimator before HMC: the machinery is sound, the inference layer is not

Prompted by a direction change: measure δt from **many correlated shower pairs** rather
than tightening one pair. That makes the model hierarchical (per-shower nuisances, one
shared δt), so the composite step stops being incidental and becomes the whole estimator.
Verifying it *before* investing in HMC is the right order, and it turned up something that
would have silently broken the plan.

### The gap that existed

Every gate in `hitman/validate` measures a trained model one ROW at a time. But a
hierarchical fit rests entirely on

    log p(x_1..x_n | Θ) − const = Σ_i log r(x_i, Θ)

which is an assumption about the DATA, not a property of the network — and nothing tested
it. Worse, the one analytic-truth test we had (`test_toy_gaussian`) trains with
`hitman.wc.toys.train_nre`, a *separate loop*. The production `fit_ratio` path had never
been compared against a ratio anyone knew.

### The ladder (`tests/test_composite_ratio.py`, 26 tests, all pass)

New: `hitman/validate/toys.py` (`HierarchicalGaussian`, closed forms audited against brute
-force MC first) and `hitman/ratio/composite.py` (grid-based composite + marginalized
hierarchical global posterior — grids because they are unconditionally correct, which is
what a *verification* instrument has to be).

| rung | claim | result |
|---|---|---|
| R1 | composite of EXACT ratios == exact posterior at ρ=0 | **0.00 %**, n=1…64 |
| R2 | at ρ>0 the sum is wrong by the PREDICTED amount | **0.00 %** |
| R3 | hierarchical global parameter, nuisances marginalized | matches theory, ∝1/√G |
| R0 | production `fit_ratio` vs the analytic ratio | 0.152 nats (see below) |
| R4 | LEARNED ratio through the proven composite | widths to **1.5 %** |
| R5 | PIT uniformity of the learned composite posterior | KS 0.021 vs 0.056 crit |

Order matters: R1 passing first is what makes an R4 failure attributable to the network.

### Finding 1 — loss and AUC cannot certify a ratio estimator

`fit_ratio` reached BCE **0.60267** against a Bayes optimum of **0.60281**, AUC 0.7093
against 0.71335 — *at the limit* — while still carrying ~0.15 nats of θ-relevant ratio
error. Chance is log 2 = 0.69315, so the entire learnable signal spans **0.09 nats**. The
objective goes flat long before the ratio is accurate. More steps/width did not help
(3000→12000 steps moved RMSE 0.32→0.25; val_loss flat from step 5000).

Consequence: on real muons, where no analytic truth exists, the ONLY instruments that can
catch this are composite calibration (R5-style PIT) and closure — never the training curve.

### Finding 2 — the error lives where the data does not

θ-relevant RMSE: **0.152** under the joint p(x,θ), 0.242 under the marginal, 0.269 on a
uniform grid. By shell: 0.090 for |θ|<0.5 (38 % of rows) → **0.506** for |θ|∈[2,2.5]
(3.1 % of rows). This is the mechanism behind the OOD guard mattering: a fit that wanders
uses a worse ratio than its loss curve ever advertised.

(An x-only additive offset would cancel in the posterior; measured, only **20–25 %** of the
squared error is of that form, so it does not excuse the rest. Hypothesis tested, rejected.)

### Finding 3 — pooling does not amplify the bias, but it does outrun it

The feared `n·b(θ)` blow-up does NOT happen — the absolute bias saturates (~0.03) because
the posterior recentres. Composite WIDTHS stay right to 1.5 % at every n.

| n | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| excess bias (learned − exact) | .016 | .024 | .029 | .029 |
| σ | .702 | .442 | .239 | .122 |
| **bias/σ** | .02 | .05 | .12 | **.24** |

A width-only check misses this entirely. Same conclusion as RL-27 reached from the other
end, now at the estimator level: **bias is the binding constraint, not variance.**

### Finding 4 — THE one that would have broken the plan: the joint MAP is biased, flat in G

**First characterization was wrong and is corrected here.** I initially described `fit_pairs`
as "profiling nuisances and reporting a profile-curvature width", and priced a
profile-curve estimator accordingly. Reading `twostation_nre.py:599-601`, it computes the
**joint MAP over all 8 parameters** and reports `sqrt([H⁻¹]_{δt,δt})` — the (δt,δt) element
of the **inverse** Hessian, which already credits nuisance correlation. That is a Laplace
approximation to the MARGINAL width, not a conditional one. The σ was never the problem.

Re-priced against exact marginalization on identical data (24 reps; MC standard error on
each bias is sd/√24):

| G | n | **A** marginal mean | **B** joint MAP (`fit_pairs`) | C profile mean | A sd | B Laplace sd | B/A |
|---|---|---|---|---|---|---|---|
| 4 | 4 | +0.085 | −0.013 | −0.174 | 0.422 | 0.537 | 1.27 |
| 16 | 4 | +0.037 | **−0.155** | −0.204 | 0.214 | 0.271 | 1.27 |
| 64 | 4 | +0.019 | **−0.180** | −0.193 | 0.106 | 0.135 | 1.28 |
| 4 | 16 | −0.030 | −0.110 | −0.342 | 0.319 | 0.421 | 1.32 |
| 16 | 16 | −0.041 | **−0.225** | −0.298 | 0.138 | 0.208 | 1.51 |
| 64 | 16 | +0.012 | **−0.206** | −0.222 | 0.066 | 0.105 | 1.60 |

* **A (marginalize) is unbiased** — every entry is within ~1.5 MC standard errors of zero.
* **B (joint MAP) is definitively biased** — −0.180 at G=64,n=4 is **8 SE** from zero;
  −0.206 at G=64,n=16 is **15 SE**. And the bias is **FLAT in G** while σ falls as 1/√G.
* **B's Laplace σ is CONSERVATIVE**, 27–60 % too wide — so the width partly masks the bias
  rather than compounding it (the opposite of what I first reported).

Bias in units of the TRUE sd — the number that decides whether stacking helps:

| n | G=4 | G=16 | G=64 |
|---|---|---|---|
| 4 | −0.03 | −0.73 | **−1.70** |
| 16 | −0.34 | −1.63 | **−3.11** |

**Stacking makes the joint-MAP estimator confidently wrong.** At G=64, n=16 the point
estimate sits 3.1 true-σ from truth (1.95σ even in its own inflated σ). The mode of a
non-Gaussian joint posterior is not its marginal mean, and that gap does not average away.
Marginalizing removes it.

**Caveat**: the non-linearity (θ + 0.45θ²) is chosen, so the −0.2 magnitude does not
transfer to the real problem. What transfers is the SCALING — MAP bias flat in G, σ ∝ 1/√G
— and the sign of the diagnosis: fix the point estimate, not the width.

### Recommendation

Proceed to HMC, but for a different reason than assumed: it is not a precision upgrade, it
is the fix for a correctness bug that only becomes visible at the scale being moved to.
Marginalizing the nuisances is what removes the bias; HMC is how you marginalize in 8+
dimensions, and it replaces a MAP point estimate with a posterior mean. The prerequisite is
met — the per-row ratio and the composite are both sound.

Note the cheap partial check available before any HMC work: the existing per-pair fit
already has everything needed to compare `dt_hat` (joint MAP) against a grid-marginalized
δt on a subset of pairs. If the two agree on real data, the toy's bias does not transfer
and stacking is safe as-is; if they diverge, the direction and size are measured before a
sampler is written.
Sequencing caution: the ~0.24 σ estimator bias at n=64 is a floor HMC will not clear; it
will give honest widths around a slightly wrong centre.

Also changed: R0's gate now measures θ-relevant error under the JOINT density. The original
raw-uniform-grid RMSE threshold (0.15) was **unreachable, not unmet** — a gate that cannot
be satisfied by a Bayes-optimal model is measuring the wrong thing.
