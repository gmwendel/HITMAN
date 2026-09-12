# run19_npe — Path B receipts (B2+B3), 2026-07-20

Training (B2): DeepSets encoder + 8-layer conditional RQ-NSF (native equinox), forward-KL
MLE on the 5M store; 40k steps in 5.5 min GPU, best val NLL 20.8310. Exports run19_npe/.

Receipts (B3, 2000 held-out events, K=256 samples, 0.73 min GPU):
- SBC: ALL 7 PARAMETERS PASS at alpha=0.01 (KS p >= 0.11 everywhere; mean ranks within
  0.007 of 0.5; edge fractions within 0.02 of nominal — no U, no hump, no slope).
  az handled circularly; t SBC valid under the coherent aug prior (N(0,50) shift applied
  to truth identically).
- TARP joint coverage: max |coverage - nominal| = 0.032 (on the diagonal).
- Posterior bias medians: x -3.8 mm, y -5.6 mm, z +5.5 mm, zen +0.001, az -0.022 (circ),
  t -0.023 ns, E +0.061 MeV. Widths: ~140 mm (xyz), 0.36 rad zen, 0.37 ns t, 0.71 MeV E.

Cross-arm significance:
- E-bias floor ~|0.06| MeV appears in BOTH constructions (ratio arm post-moment-fix -0.06;
  NPE +0.06 with NO penalties) => likely a genuine information/prior floor, not estimator
  artifact.
- First fully calibrated posterior of the campaign, from the normalized-by-construction +
  MLE arm, at ~1/100th the training cost of the ratio arm — empirical confirmation of the
  normalization-difficulty analysis (D2 map: e^{D2} ~ 1e80-1e304 prices any
  reference-measure normalization estimator; the density arm never pays it).

Deferred: 200-event NUTS spot-check until Path A (exact-quadrature rework) retrains.
Artifacts: run19_npe/{sbc_tarp.png, sbc_results.npz, receipt_sbc.log},
training_runs/receipt_run19_sbc.py.
