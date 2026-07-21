# Composable SBI API — migration notes

HITMAN's numeric core (closed-form densities, identity/receipt machinery, conditional
flows) is detector-agnostic; what used to leak was the water-Cherenkov (WC) *layout* — the
7-parameter hypothesis `(x, y, z, zen, az, t, E)`, the per-hit `(sensor xyz, time)` mark, the
discrete sensor grid, and the monolithic splinemle likelihood. Two small, additive protocol
layers lift that layout out of the core so a new detector plugs in without editing core
modules. **Everything below is backward compatible: the WC path is one instantiation of each
protocol, existing imports and the run18/run19/run20 entry points are unchanged.**

---

## Proposal 1 — detector-agnostic Obs/Hyp spec (`hitman/spec.py`)

Two injectable value objects (pure metadata — NamedTuples of scalars/strings/tuples, never
threaded through `jit`/`vmap`):

```python
class DimSpec(NamedTuple):
    name: str
    lo: float | None = None      # prior box (box dims only)
    hi: float | None = None
    circular: bool = False       # periodic (azimuth, signed shower angle); uses `period`
    positive: bool = False       # physically > 0 (energy, width)
    cos: bool = False            # polar angle in [0, pi], handled in cos() space by the flow
    period: float = 2*pi
    units: str = ""
    @property
    def kind(self) -> str        # 'box' | 'cos' | 'circ'  (flow-domain dispatch)

class HypSpec(NamedTuple):
    dims: tuple[DimSpec, ...]
    zenith: int | None = None    # index used by direction()
    azimuth: int | None = None
    # derived views (consumed by the flow / receipts / moments):
    dim, names, kinds, box, cos_dims, circular_dims, positive_dims, period
    def index(self, name) -> int
    def direction(self, theta) -> jnp.ndarray   # unit 3-vector from (zenith, azimuth)

class ObsSpec(NamedTuple):
    mark_names: tuple[str, ...]                 # columns of EventBatch.hits (continuous mark)
    count_names: tuple[str, ...] = ("total_charge", "n_hits")   # columns of EventBatch.charge
    has_sensor_index: bool = False              # whether EventBatch.pmt_id (discrete mark) is used
    circular_marks: tuple[int, ...] = ()
    @property
    def n_marks(self) -> int
    def index(self, name) -> int
```

**Provided instances.** `WC_HYP_SPEC` / `WC_OBS_SPEC` reproduce the exact WC layout (locked to
`nn.features`, `npe.flow`, `train.moments` by `tests/test_spec.py`). `MUON_HYP_SPEC` /
`MUON_OBS_SPEC` are a deliberately different instantiation (continuous angular marks
`(dt_plane, alpha_r, alpha_t)`, no sensor grid, profile-summary hypothesis
`(cos_zen, Xmu_max, width, X1, r, psi)` with a signed circular `psi`).

**Wiring (all backward compatible).**
- `hitman.npe.flow.ConditionalFlow.from_spec(spec, context_dim, *, key, **kw)` — derives
  `n_dim`/`box`/`cos_dims`/`circular_dims`/`period` from a `HypSpec`. WC default is exactly
  `from_spec(WC_HYP_SPEC, ...)`.
- `hitman.receipts.schema.validate_schema(receipts, hyp_dim=None)` — expected `truth` length
  is `hyp_dim`, else `receipts["meta"]["hyp_dim"]`, else 7 (WC). Record `meta.hyp_dim` for a
  non-WC detector; WC receipts are untouched.
- `hitman.train.moments` — `_P`/`VECH_IDX` dimension now come from `WC_HYP_SPEC.names` (same
  values). *Deferred:* the ray-projection instrument (`direction`, `_ray_entries`) is still
  the WC `(zen, az)` ray; making the projection instrument injectable is DESIGN proposal 4.
- `hitman.data.structures.EventBatch` — unchanged fields; new `n_hyp`/`n_marks` properties and
  a spec-driven layout contract in the docstring. A `(n_events, K)` hyp and `(n_hits, M)` mark
  already fit with no structural change.

**To onboard a detector:** define a `HypSpec`/`ObsSpec`, build the flow with `from_spec`, and
set `meta.hyp_dim` in receipts. Nothing in the core needs editing.

---

## Proposal 2 — pluggable density-factor heads (`hitman/density/`)

A per-event likelihood composes as a product of exactly-normalized factors:

```
log L(event | context) = count.log_prob(N | context)
                       + sum_objects [ sum_marks mark_k.log_prob(v_k | context) ]
```

Every factor normalizes in closed form (no Monte Carlo, no grid in the data dimension), so
any product is itself a proper density.

```python
@runtime_checkable
class DensityFactor(Protocol):
    def log_prob(self, obs, *context): ...   # exactly-normalized log density/pmf of one obs

class LogSplineFactor(eqx.Module):           # continuous 1-D mark (piecewise-exp log-spline)
    def __init__(self, knots, floor=-inf)
    def log_prob(self, u, node_vals, floor=None)   # node_vals = per-object shape context
    def log_partition(self, node_vals)             # exact closed-form log Z
    def density(self, u, node_vals)

class SoftmaxMarkFactor(eqx.Module):         # discrete mark (WC sensor); stateless
    def log_prob(self, s, eta)                     # eta[s] - logsumexp(eta)
    def log_partition(self, eta)                   # logsumexp(eta) = log mark intensity

class CountFactor(eqx.Module):               # NB2/Poisson total count + monotone yield head
    def __init__(self, *, phi_knots, phi_init_values=None, phi_offset=241.0,
                 phi_anchor=None, disp_init=(3.5, 0.15), count_model="nbinom", x_ref=5.0)
    def phi(self, x); def log_dispersion(self, x); def phi_values
    def log_mean(self, log_intensity, x)           # log Lambda = phi(x) + log_intensity
    def log_count(self, N, log_Lambda, x)
    def log_prob(self, N, log_intensity, x)        # DensityFactor entry point

def marked_poisson_loglik(count_ll, mark_lls, mask=None)   # count_ll + sum(mark_lls*mask)
```

Pure closed-form functions are also exported for direct use: `log_expm1_over_x`,
`interval_log_integrals`, `logZ_time`, `ell_at`, `log_prob_u`, `density_u` (logspline);
`nb2_log_count`, `poisson_log_count`, `monotone_phi_values`, `phi_at`,
`affine_log_dispersion`, `init_phi_params` (count).

**Backward compatibility.** `hitman.splinemle.model` imports these from `hitman.density` and
**re-exports the historical names**, so `from hitman.splinemle.model import log_prob_u,
SUPPORT_FLOOR, ell_at, ...` and the `hitman.splinemle` package exports are unchanged.
`SplineMLE` keeps its own trainable leaves and calls the same pure functions, so the
monolithic model is **exactly reconstructible** from `SoftmaxMarkFactor x LogSplineFactor x
CountFactor` (verified bit-for-bit to <1e-9 in float64 by
`tests/test_density_factors.py::test_wc_model_reconstructed_from_factors`).

**Downstream composition example (muon chain-of-conditionals):**
```python
# NB2(N | Theta) x f(dt|Theta) x f(alpha_r|dt,Theta) x f(alpha_t|dt,alpha_r,Theta)
count = CountFactor(phi_knots=(...), count_model="nbinom", x_ref=...)   # yield on N
dt_f  = LogSplineFactor(knots=asinh_warped_dt_knots)                    # root continuous mark
ar_f  = LogSplineFactor(knots=alpha_r_knots)   # or an RQ-spline factor (npe.flow.*_rqs)
# node_vals for each factor come from your per-station conditioner net given (dt, Theta).
mark_ll = dt_f.log_prob(dt, dt_nodes) + ar_f.log_prob(a_r, ar_nodes(dt, Theta)) + ...
event_ll = marked_poisson_loglik(count.log_prob(N, log_intensity=0.0, x=Theta_x),
                                 per_object_mark_lls, mask)
```
No new normalization code is written: each factor owns its exact partition.

---

---

## Proposal 3 — reusable `(model, data)` receipt harness (`hitman/receipts/harness.py`)

The numeric receipt core (`chi2`, `reweight`, `forward`, `score`, `schema`) was already
detector-agnostic and synthetic-net tested; the water-Cherenkov coupling lived in
`runner.py` (z-ring strata, marginal-pool reweighting, the NRE model interface). Two seams
lift it out:

```python
@runtime_checkable
class ReceiptModel(Protocol):                       # what the battery needs from a model
    def pool_log_ratios(self, pool_pmt, pool_t, pmt_pos, theta) -> np.ndarray: ...
    def count_grid_logits(self, n_grid, theta) -> np.ndarray: ...

class NREReceiptModel(ReceiptModel):                # WC hitnet + chargenet instantiation
    def __init__(self, hitnet, chargenet, obs_style="xyz", chunk=500_000)

def percentile_strata(values, n) -> (labels, n_labels)   # detector-agnostic strata primitive
def z_rings(pmt_pos, n_rings) -> (labels, n_labels)      # WC default = percentile_strata(z, n)

def compute_forward_block(..., strata_fn=None)           # strata_fn defaults to z_rings
def forward_block_from_model(model, pool_pmt, pool_t, pmt_pos, mc_pmt, mc_t, mc_ntot,
                             n_mc_events, n_grid, train_n_hist, theta, *,
                             n_rings=8, strata_fn=None, time_bins=None, ring_time_bins=None)
```

**WC path is one instantiation.** `runner.run_model_dir` now wires `NREReceiptModel` + the
default `z_rings` through `forward_block_from_model`; `compute_forward_block` / `z_rings` /
`hit_logits` / `chargenet_logit_grid` are re-exported from `runner` so historical imports are
unchanged. `tests/test_receipts_harness.py` locks the model-protocol path to the direct
`compute_forward_block` numbers bit-for-bit.

**Shower-frame `E × r` strata (the muon need).** The score/Bartlett receipts are already
caller-stratified — `score.score_identity(grad_fn, xs, names=...)` takes the component names
(supply proposal-4 instrument names), and `forward.per_ring_time_receipt` takes *precomputed*
ring labels. So a downstream project runs the same battery by (a) passing a `strata_fn` that
bins on core distance `r` (e.g. `lambda pos, n: percentile_strata(radius(pos), n)`), and
(b) calling `score_identity` per `(E × r × sign ψ)` stratum with instrument-projected names.
No core edit is required.

---

## Proposal 4 — projection-instrument injection (`hitman/train/moments.py`)

```python
class ProjectionInstrument(NamedTuple):
    project: Callable                 # theta -> unit vector over `spatial`
    spatial: tuple = (0, 1, 2)        # parameter block the projection spans
    partners: tuple = (5, 6)          # scalar params whose cross-coupling is reported
    partner_names: tuple = ("t", "E")

RAY_INSTRUMENT = ProjectionInstrument(project=direction, spatial=(0,1,2),
                                      partners=(5,6), partner_names=("t","E"))  # WC default

def lean_vector(g, H, theta, instrument=RAY_INSTRUMENT)
def full_vector(g, H, theta, instrument=RAY_INSTRUMENT)
def moment_names(names=None, instrument=RAY_INSTRUMENT) -> (LEAN_NAMES, FULL_NAMES)
```

Implemented as **injection-with-default**: the default WC path reproduces the pre-refactor
hardcoded `(zen, az)` ray formulas **bit-for-bit** (verified over float32 and float64 by
`tests/test_moments_instrument.py::test_default_lean_full_bit_identical_to_preRefactor`), so
the calibrated WC receipts and their frozen `W` are untouched. A downstream project injects a
shower-axis instrument with its own `partners`/`partner_names` (e.g. `("dt", "Xmax")`) to
turn "project onto the ray" into "project onto any instrument".

---

## Proposal 5 — encoder feature-map callable (`hitman/npe/encoder.py`)

```python
DeepSetsEncoder(*, key, ..., feature_map=_hit_feature_map, n_hit_in=N_HIT_IN)
```

`feature_map` (a static field, WC default `_hit_feature_map` → 5 columns) maps padded hit
rows to normalized per-hit features; `n_hit_in` is its width. Inject a different pair to
encode a non-WC mark (e.g. continuous `(dt, alpha_r, alpha_t)`). The default with an explicit
WC map is identical; a custom map round-trips through `eqx` serialization (the map rides the
template, like the MLP activation strings).

---

## Deliberately deferred (with reasons)

- **`SplineMLE` was not decomposed into held sub-factors** (proposal 2, unchanged decision).
  Moving `phi_e0`/`phi_raw`/`disp` into a `CountFactor` submodule would break the existing
  `m.phi_e0` / `grad.phi_raw` test contract. The reusable factors live in `hitman.density` and
  `SplineMLE` calls the same pure functions — same numbers, no API break.
- **`npe.flow.DEFAULT_BOX`/`COS_DIMS`/`CIRCULAR_DIMS` literals** are left in place (a
  `ConditionalFlow.from_spec` builder + a lock test tie them to `WC_HYP_SPEC` rather than
  rewriting the literals) so no WC caller importing those constants changes.
- **The WC forward *observable set* (per-PMT charge, z-ring time) was made strata- and
  model-pluggable but NOT unified with continuous-mark observables.** The `per_pmt_charge` /
  `per_ring_time` receipts are intrinsically sensor-grid observables; a continuous-mark
  detector (muon `(r-annulus, dt)`) defines its *own* forward observables and reuses the
  shared seam — the reweighting + chi2 + strata core (already generic) plus the `ReceiptModel`
  / `strata_fn` protocol. Forcing physically different observables through one abstraction
  would add speculative surface and risk the calibrated WC numerics for no concrete gain;
  deferred until a second detector's forward observables exist to generalize *against*.
- **`runner.py` I/O (`load_models`, `build_marginal_pool`) stays WC/HitStore-specific.** These
  are the model-class and store loaders, not battery logic; a downstream detector supplies its
  own loaders and calls `forward_block_from_model` / the numeric core directly.
```
