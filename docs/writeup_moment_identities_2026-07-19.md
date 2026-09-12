# Moment identities for neural likelihood-ratio reconstruction

*A field report: what exact score constraints fixed in a HITMAN-style surrogate, what they
revealed next, and where the mathematics says to go. Written for a reader familiar with the
approach of Eller et al., arXiv:2208.10166. All numbers are from our JAX reimplementation
(60-sensor water-Cherenkov geometry, 5M simulated e⁻ events, 0–10 MeV, 7-parameter
hypothesis θ = (x, y, z, zen, az, t, E)).*

---

## 1. Setup and the symptom

We train the standard two-network surrogate from the paper: a **hitnet**
r̂_hit(hit, θ) and a **chargenet** r̂_chg(N, θ), each trained as a binary classifier
between matched pairs (x, θ_true) and shuffled pairs (x, θ'), so that the logit converges
to the log likelihood-to-evidence ratio. Event log-likelihood = Σ_hits log r̂_hit + log r̂_chg;
reconstruction = gradient-based MLE / NUTS posterior over θ on this surface.

The networks pass every conventional check. Per-hit validation BCE is at its plateau;
forward-model receipts are excellent (reweighting the training marginal by r̂ reproduces
per-sensor time distributions over four decades, ring by ring). And yet the reconstruction
has a large coherent systematic: on 50k prior-draw events the fitted energy reads
**ΔE median = −0.60 MeV** with a heavy low-side shoulder, growing to −0.9 MeV and worse in
joint fits at high E ("the E-wall"). Truth-anchored tests (binned-MC likelihood on dedicated
anchor samples) showed the *exact* likelihood is nearly unbiased in E at fixed position —
the bias is a property of the surrogate, paired difference −0.437 ± 0.002 MeV at 8 MeV.

## 2. Why "BCE converges to the true ratio" does not protect you

The classifier loss is a strictly proper scoring rule: its **population** minimizer is the
true ratio. Three things break in practice, and they compound:

1. **Metric mismatch (the essential one).** BCE controls an L²-type distance on (x, θ)
   jointly. Inference consumes something else entirely: **θ-derivatives of the
   event-summed log-ratio at fixed x**. Differentiation is unbounded — closeness in the
   BCE metric implies nothing about closeness of ∇_θ log r̂.
2. **Coherent amplification.** A per-hit tilt of 10⁻³ nats/hit is invisible to per-hit
   classification (it moves each classification probability by <0.1%), but the fitter sums
   it over ~100 hits and differentiates. Small-and-coherent beats the loss; inference pays.
3. **Wasted capacity.** Any additive function c(x) of the observation alone changes the
   BCE but not the reconstruction (it cancels in θ-differences). The loss spends capacity
   on a direction inference never reads, while underweighting θ-slopes, the only direction
   inference does read.

So the trained network is one member of a large family of near-BCE-optimal functions, and
BCE has almost no opinion about which member you get *in the directions that matter*.

## 3. The first moment: score identity as an exact constraint

For any TRUE ratio r = p(x|θ)/p(x), the classical score identity holds at every θ
(the reference p(x) is θ-free):

**E_{x ~ p(·|θ)} [ ∇_θ log r(x, θ) ] = 0.**

A learned ratio violates it by a coherent mean tilt **b(θ) = E[∇_θ log r̂]**, measurable on
held-out simulation with nothing but autodiff — no truth densities needed. Measured on the
production model, b_E(θ) is a smooth wall: z-scores from +7.5 at threshold to **−224** at
9–10 MeV, and the first-order drift prediction H⁻¹b reproduces the observed MLE biases in
sign and structure. The E-wall is a quantified violation of an exact identity.

We then add the violation to the training loss. The estimator design matters more than the
idea:

- **Stratify in true E** (fixed 1-MeV bins): b(θ) is θ-dependent and flips sign at the
  Cherenkov threshold; a global mean washes it out.
- **Split-half cross product.** Per-event scores fluctuate at the *Fisher* scale — that
  scatter is information, not error. The penalty per stratum is ⟨ḡ_A, ḡ_B⟩ over a random
  half-split of the batch: an unbiased estimator of ‖E[g]‖² in which per-event noise
  cancels in expectation. Penalizing mean ‖g‖² instead would shrink the Fisher information
  itself — the one failure mode this construction provably avoids.
- Per-stratum whitening (score scales span ~8× across strata); winsorization at measured
  P99.9 (sparse time-tail robustness; receipt: ≤0.4 SE induced shift).
- Weight λ calibrated once at init so the penalty gradient is ~7.5% of the BCE gradient;
  scanned {λ*, 3λ*} against the λ=0 control.

**Results (same events, same solver, control vs constrained):**

| quantity | control | constrained |
|---|---|---|
| b_E drift H⁻¹b (prior-avg) | −1.47 MeV | −0.02 MeV |
| 50k ΔE median | −0.60 MeV | **−0.06 MeV** |
| 50k ΔE IQR/1.35 | 1.10 | 0.69 (−37%) |
| 50k ΔE RMS | 1.42 | **0.85 (−40%)** |
| all other params (bias, width) | — | unchanged (≲1%) |
| corr(ΔE, Δ_along-ray), 6–10 MeV | +0.13 | **+0.01** |
| forward receipts (ring time shapes, occupancy) | — | unchanged; bright-ring intensity error −4.7% → −1.0% |
| truth-anchored paired E bias @8 MeV | −0.437 | −0.161 |

The key qualitative point: **the width fell *with* the bias.** The tilt was E-dependent, so
different true energies were shifted by different amounts — that spread masqueraded as
resolution. Removing a coherent, θ-dependent bias removes its variance contribution too.
The error was removed, not relocated: every other marginal, the angular resolution, and all
data-space distributions are invariant. Cost: a few percent of training time.

## 4. What the first moment could not fix — and how we know

Two residuals survived, and the λ-scan is the tell: tripling λ drives the enforced moment
even closer to zero but does **not** improve the downstream residuals (one got worse).
They are structurally different objects:

**(a) A direction-correlated first moment.** Projecting the spatial score onto each
event's own direction, s_ray = E[(∇_xyz log r̂)·d̂], reveals a residual tilt (~−0.015,
E-independent) that detector-frame strata *cannot see*: under an isotropic prior it
averages to zero in every stratum. In the control model s_ray grew with E (−0.035 at
10 MeV); the constraint removed the E-tilt-in-disguise half and left the genuinely
ray-aligned half. This residual matters beyond position: the 50k residuals show
**corr(Δt, Δ_along-ray) = 0.85** (the time-of-flight degeneracy), so the surviving joint-fit
t bias (+0.12 ns) is this spatial tilt wearing a time costume — the *conditional* t-profile
is separately verified near-exact against truth (within 15–26 ps; the true likelihood itself
carries +31 ps of intrinsic finite-sample mode bias from its causally sharp rising edge).

**(b) A second-moment miscalibration.** The next identity in the tower is Bartlett's:
**E[∇²_θ log r + (∇_θ log r)(∇_θ log r)ᵀ] = 0** — curvature must equal score covariance
(both equal the Fisher information). Testing the E-axis against cached truth-anchored
likelihood profiles (paired, per event): the surrogate's implied σ_E is **14–17% too wide**
(0.80 MeV true → 0.89–0.92 surrogate, ~19σ paired) and the profile is **under-skewed**
(cubic coefficient +0.195 true vs +0.09–0.13 surrogate). Both are λ-independent — a
different moment, untouched by the first-moment constraint, and they account for the
remaining truth-anchored E bias (a flatter, less-skewed profile pulls the posterior mean
low even when the mean score is exactly zero). The zeroth-order symptom of the same
disease: the surrogate's self-normalization Z(θ) = E_{p(x)}[r̂] reads 1.28–1.35 where the
true ratio gives identically 1.

**Covariance checks belong in the battery.** The off-diagonal entries of the second moment
set the error-ellipse orientation and the degeneracy geometry (the "cone drive" is an
E–position off-diagonal phenomenon). Two lessons from doing it: (i) it must be done in the
**event's ray frame** — detector-frame cross-correlations average out under isotropy;
(ii) it caught a real improvement invisible to every marginal check (the E×along-ray error
decorrelation above) and localized the dominant remaining coupling (t×along-ray).

## 5. What are we actually doing? (the clean abstraction)

Inference only ever uses θ-sections of log r̂ at fixed x, modulo c(x) — equivalently, the
shape of the per-event posterior p̂(θ|x). The natural deployment metric is therefore

**L* = E_x [ KL( p(θ|x) ‖ p̂(θ|x) ) ].**

Every identity above is a Taylor coefficient of L* in the error field: the score identity
is its gradient content, Bartlett its curvature content, the covariance blocks its mixed
partials, with Z(θ) as the generating normalization. Stacking moment penalties = assembling
the Taylor expansion of L* by hand, one λ each. The natural question (raised, correctly, as
an objection): why not train on the exact object? L* is the population limit of the
**contrastive / ranking form of NRE** (NRE-C: softmax classification of θ_true among K
contrast hypotheses drawn from the prior) — λ-free, and its optimum controls everything at
once.

The answer is a finite-sample information argument, not a taste preference. Both losses
estimate the same violation coefficients from the same events through different channels:

- The **derivative channel** (moment penalties) reads ∇_θ log r̂ and ∇²_θ log r̂ per event
  *exactly*, by autodiff. Estimating a coherent tilt this way operates at the Cramér–Rao
  bound for that functional. No bandwidth, no design knob.
- The **label channel** (NRE-C) must infer the same coefficients from K-way label
  statistics. Its per-contrast Fisher information carries the softmax-variance factor
  w(1−w), which collapses in both regimes: contrasts much tighter than the posterior give
  information O(δ²) (and O(δ⁴) for curvature); contrasts much wider give
  w(1−w) ≈ exp(−Δlog L) → 0. With prior-drawn contrasts and our posterior scales, the
  7-dimensional useful-volume fraction is ~10⁻⁵ — an effective-sample deficit of 10³–10⁴
  versus the derivative channel, before paying K× compute. The data-processing inequality
  makes the ordering strict: the label is a stochastic quantization of the very quantities
  autodiff reads directly.

So: **prior-contrast NRE-C ≪ moment penalties** on the shared (coherent, low-order)
coefficients — but the exact form uniquely owns the *incoherent* content (x-varying shape
error, which adds estimator variance rather than bias, plus tails/multimodality). The
constructive reconciliation, if the exact form is ever needed: draw contrasts from the
model's own per-event Laplace ellipse N(θ̂, c·Ĥ⁻¹) with the proposal correction in the
softmax — the machinery (per-event Hessians by AD) already exists, w(1−w) becomes O(1) by
construction, and the loss is population-exact and λ-free at K× cost. Whether that content
matters is itself measurable: compare achieved residual covariance against the truth-implied
Cramér–Rao bound.

## 6. Distilled takeaways

1. A surrogate can pass per-hit validation and forward-model checks and still carry an
   inference-breaking systematic: the training loss and the inference functional live in
   different topologies. Check the functionals inference uses.
2. Exact identities of the true ratio (score, Bartlett, normalization) are free,
   truth-independent diagnostics — and, with a variance-respecting estimator (stratified
   split-half), safe training constraints: their population optimum *is* the original
   target, so no bias is introduced at any λ.
3. First moment enforced ⇒ coherent location bias dies, and (nontrivially) so does its
   variance contribution: our E error fell 40% in RMS with everything else invariant.
4. The residuals after the first moment are informative by construction: a
   direction-correlated tilt (add the ray-projected component — the identity holds for any
   θ-dependent projection E[g·v(θ)] = 0) and a curvature/skew miscalibration (Bartlett
   territory: penalty, or post-hoc recalibration, or adaptive-contrast training).
5. Do covariance checks, in the event's ray frame: the degeneracy geometry is
   off-diagonal, and prior-averaged detector-frame statistics are blind to it.
6. The moment stack is the Taylor expansion of the per-event posterior KL; contrastive
   NRE is its generating object. The choice between them is a computable
   information-per-sample trade (softmax-variance vs autodiff), not an aesthetic one —
   and with prior contrasts at sharp-posterior scales, the exact form is information-starved
   precisely where the miscalibration lives.
7. If moments beyond second/third are ever needed, stop: that's the estimator saying the
   architecture fights the physics, and a structural parameterization (e.g. factorizing the
   energy response into a monotone yield curve × geometric collection efficiency — both
   independently measurable from the simulation's emission truth) is the better spend.

*Artifacts: implementation `hitman/train/identities.py` (v2.0-jax, commit 2185b8a); full
receipt battery and per-figure evidence in `training_runs/run12_scoreid_5M/` and
`director/report_scoreid_run12_2026-07-19.md`.*
