# Two-Path Posterior Cross-Check (handoff plan of record, 2026-07-20)

User-supplied director-level plan. NAMING NOTE: the plan's "Path A — run15" collides
with the existing run15_finetune (2026-07-19 warm-start GMM fine-tune); Path A is
implemented as **run18_nwj**, Path B as **run19_npe**. Everything else verbatim.

## Path A — run18_nwj: normalized-tilt MLE with jointly trained Znet (NWJ form)
A1. Joint objective L(phi,psi) = -E_matched[f_phi - a_psi(theta)]
    + E_shuffled[exp(f_phi - a_psi(theta))] - 1. Same batch plumbing as BCE.
    Winsorization on the shuffled exponential tail. Warm-start f from run12 (lam1),
    a_psi from an IS/quadrature estimate of log Z at those weights; faster LR (or EMA
    target) on psi than phi. Chargenet unfrozen, exact 1-D partition sum.
A2. Toys before training (CPU): exact-ratio (gradient vanishes, a_psi -> log Z);
    deflation toy (restoring force = E_data[s] - E_model[s]); bound-gap toy
    (E_shuffled[e^{f-a}] -> 1 per stratum).
A3. Train run18_nwj (single GPU job, same seeds/data/rails as run10/12/14). In-training
    gates: response receipt Jhat/Jhat_ctrl in [0.9, 1.1] per stratum; bound-gap
    per stratum -> 0.
A4. Receipts (offline): quadrature slow-path check of a_psi (<=1e-4 nats sup);
    bank-split test on the marginal; full White battery on f - a_psi; 50k MLE battery.
    Pre-registered vs run12: truth-anchored E bias @8 MeV, dE median/RMS, s_ray,
    Bartlett-E honesty, profile skew.

## Path B — run19_npe: forward-KL conditional flow
B1. DeepSets encoder (per-hit features -> shared MLP -> sum-pool + count channel ->
    context); conditional NSF (~8-10 layers) over 7-D theta. Log summary width
    (sufficiency risk).
B2. Train on the same 5M joint pairs, plain -log q(theta|x). Single GPU job.
B3. Receipts: SBC per parameter + TARP joint coverage on held-out; spot-check ~200
    per-event posteriors vs run18 NUTS.

## Path C — cross-receipt
C1. Disagreement map Delta(x,theta) = [log q(theta|x) - log pi(theta)] -
    [f(x,theta) - a(theta)] on ~50k held-out events over each event's Laplace-scale
    theta-grid; mean/gradient/curvature of Delta per E-stratum and in the ray frame.
C2. NPE as NUTS init/proposal for run18; quiet map where physics lives = headline
    correctness certificate.

## Sequencing / stop conditions
A1+A2 || B1 (dev, CPU) -> A3 (GPU) -> B2 (GPU) -> A4/B3 (offline) -> C.
Path A Bartlett block significant after clearing everything else => model-class floor
=> structural yield factorization (task #8), do not iterate constraints.
Path B SBC failure => widen flow/summary before touching Path A.
Disagreement structured at posterior cores => halt, bisect with anchor battery.

## Resource rules
Opus subagents do all task work; director reviews gates. ONE training job on GPU at a
time; no concurrent device probes; receipts/toys CPU-side. 450 W cap, cgroup scopes,
capture guard, big-arrays-as-jit-args, mkdir-before-redirect, no pgrep self-match.
Subagents declare estimated VRAM/step-time before device jobs; no speculative queuing;
one W/holdout re-estimation pass in budget only if gates demand.
