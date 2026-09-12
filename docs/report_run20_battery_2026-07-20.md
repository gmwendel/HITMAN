# run20 splineMLE — receipt battery (2026-07-20)

Model: closed-form conditional density (log-spline t in TOF-residual u, 23 knots;
241-sensor softmax + Poisson total tied to one eta; n_eff learnable -> 1.134, weakly
identified vs conditioner d-dependence). Trained 43.5 min GPU, best val 10.68945.

## Verdicts
1. FIRST-MOMENT HONESTY BY CONSTRUCTION — PASS: b_E |z|<=3.6 all strata, s_ray ~ 0,
   with ZERO penalties (vs lam1's raw b_E z range -224..+7.5 pre-fix). The
   architectural thesis receipted on a trained model.
2. BARTLETT — FAIL, MECHANISM NAILED: g_E^2 / (-H_EE) = 2.0->4.6, tracking the data
   Fano (3.7->7.4 rising with E) stratum-for-stratum. The Poisson count model's
   overdispersion inflates count-score variance while its Hessian asserts Fano=1;
   the second-moment dishonesty IS the NB2 gap, quantified. Naive -H^-1 E-errors too
   tight; sandwich or NB2 needed.
3. Forward ring: peak position exact (5.50 vs MC 5.57ns), peak 12% smoother (0.630
   vs 0.721; 23-knot smoothness), zero support leakage; ring rate over-predicted 25%
   (0.337 vs 0.2696/sensor) — conditioner/intensity allocation, watch in v1.
4. Five-point CE_excess: -3.19/-3.47/-3.22/-4.76/-3.14 vs lam1 -3.45/-3.57/-3.50/
   -4.81/-3.25 — 90-95% of lam1's reduction everywhere incl. the wall. Competitive v0.
5. Paired eridge: -0.185 +- 0.003 MeV (run10 -0.437, lam3 -0.284, lam1 -0.161,
   target |<=0.1|). E-wall persists at reduced size — E enters only as the (E-1)
   conditioner feature; phi(E) yield head still queued.

## v1 actions (each named by a receipt)
- NB2 count term (one trainable dispersion; deferred hook exists) -> receipts 2,5,
  plausibly 4 (count-score variance pollutes E-inference).
- phi(E) monotone yield head on the intensity (harvested log Ybar(E) init) -> receipt 5.
- Intensity allocation check / more knots or width if ring rate stays high -> receipt 3.
Artifacts: run20_splinemle/{ring_forward.png, moments.npz, ce_fivepoint*.npz,
eridge_run20.npz, fano.npz, receipt scripts + logs}.
