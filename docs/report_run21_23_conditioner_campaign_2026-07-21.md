# run21→23: the sensor-allocation campaign (2026-07-20/21)

Arc: three receipted falsifications converged on a structural fix worth +1.75 nats/event of
true likelihood. Full detail in the campaign director reports; this is the transferable story.

## The chain
1. **run21 (distance-basis features, 5→9 invariants, width 256/depth 4):** INERT. Features
   demonstrably wired, allocation unchanged (TV(v1,v2)=0.01), ring excess +25% untouched.
   Lesson: a binned-residual regression (R²=0.86) does NOT imply per-event fixability —
   softmax coupling and the training objective decide, not feature-space predictability.
2. **run22 (split eta/time trunks):** NULL. A dedicated fresh eta trunk converged to the
   same allocation to 4 decimals — refuting "shared-trunk gradient starvation".
   Lesson: two architectures finding the same optimum means the OBJECTIVE owns the gap.
3. **Coupling test (decisive, cheap, CPU):** logΛ = phi(E) + logsumexp(eta) couples
   allocation shape to intensity. The sensor-optimal correction shifts logsumexp by an
   E-orthogonal residual (+0.164 nats/evt after best phi repair) that the loss — which
   weights each event's count factor ~mean_N× more than its sensor factor — prices out.
   Directional derivatives: sensor-NLL descends along the correction, joint loss climbs.
   NOTE: the TRUE per-event likelihood favors the correction (−0.97 nats/evt); the
   as-implemented per-hit/per-event loss does not. Receipts evaluate the true likelihood.
4. **run23 (decoupled intensity head):** logΛ = psi(6 O(2)-invariants of θ) as its own head,
   eta pure shape (gauge-centered), psi DISTILLED to v2's intensity surface before joint
   training (acceptance gates: |ΔlogΛ| std ≤0.05, count NLL within 0.02 of anchor — two
   distill iterations needed: 2-feature psi missed direction dependence; then a scalar bias).
   RESULT: sensor −0.0179/hit (beats the frozen-correction ceiling 0.0118), count −0.0267/evt
   (psi beats phi+lse as intensity model), time held, val −0.041 on a common measure.

## Closing battery highlights
Allocation KL down everywhere: symmetric probe 0.097→0.080, transverse probes −11%,
off-center ball 0.059→0.049. CE five-point better at all points. Bartlett strata UNCHANGED
(the hit-factor score over-dispersion is the loss/receipt objective mismatch — next lever:
train on the true per-event NLL). eridge −0.175 (E-response calibration, separate lever).
Axial ring +18% persists BY DESIGN: population-optimal invariant models decline corrections
at measure-poor, likelihood-conflicting corners (receipted via a dedicated sensor-only fit
that recovered population likelihood while refusing the axial fix).

## Symmetrization discipline (important for any symmetric detector / shower surrogate)
Symmetry-averaged receipts shield over-symmetrized architectures: an O(2)-invariant model is
exactly constant on sensor orbits, so orbit-mean comparisons cannot detect missing asymmetric
structure. Required receipts: (a) MC within-orbit scatter vs Poisson (here: real 3-4×
super-Poisson structure exists, but its irreducible KL floor is 0.15% of the model's
allocation KL — symmetry is NOT the wall); (b) asymmetric probe points compared per-sensor,
unaveraged (here: within-orbit fraction of residual variance ≤0.002 — the remaining error is
between-orbit, i.e. representable, an optimization/loss issue not a symmetry limit).

## Estimator-level cross-check
Four point estimators on the same 5k prior-draw events: run23's exact-likelihood multistart
MLE (NPE-amortized init, 33 ms/event on GPU) matches the best NRE's vertex/timing core,
beats all estimators on direction, and cuts timing failure tails ~30% vs cold multistart.
Its weak axes (z bias +15 mm, E width) are exactly the battery's residual model defects
expressing as estimator symptoms — the receipt battery predicts estimator behavior.

## Operational landmines (new this campaign)
- Harness background tasks die at ~60 min: launch long trainings as DETACHED transient
  systemd services (no --scope), driver tees its own log, poll via systemctl.
- Driver val conventions drift across runs: cross-run val comparisons need a common-measure
  eval (same loss code, same events). Within-run checkpoint selection is unaffected.
- Distillation inits need acceptance gates (they caught two real spec errors here before
  any GPU time was spent).
