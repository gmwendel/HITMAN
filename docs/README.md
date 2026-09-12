# Campaign findings (v2.0-jax, 2026-07-19/20)

Key documents for reusing this stack as a surrogate-model framework (e.g. for
CORSIKA muon-shower surrogates):

- `writeup_moment_identities_2026-07-19.md` — why BCE/NRE-trained ratios fail
  inference (metric mismatch), the exact-identity diagnostic/penalty program, and
  its receipts. The Renyi-2 difficulty map D2(theta) = N·d2/hit prices any
  reference-measure normalization estimator (e^{D2} ~ 1e80-1e304 here) — compute
  it FIRST on any new detector/observable from two histograms.
- `plan_twopath_2026-07-20.md` — the two-path cross-check design (normalized-tilt
  MLE arm x NPE arm x disagreement map).
- `report_run19_npe_2026-07-20.md` — NPE arm (hitman/npe): DeepSets + conditional
  RQ-NSF, SBC-calibrated on all 7 params in 5.5 GPU-minutes.
- `report_run20_battery_2026-07-20.md` — closed-form density arm
  (hitman/splinemle): log-spline time density in TOF-residual coordinates,
  softmax sensors, Poisson/NB counts; first-moment honesty BY CONSTRUCTION;
  every residual receipt names a physics fix.

Headline transferable lessons: (1) normalization must hold identically in the
forward pass — it cannot be enforced by sampling in any variable; (2) train on
the deployment metric (exact MLE of a tractably-normalized density) and the
identity tower becomes free; (3) keep a receipt battery (moments, truth-anchored
profiles, forward shapes at matched resolution, SBC/TARP, cross-arm disagreement
maps) — every failure above was caught, attributed, and named by a receipt.
