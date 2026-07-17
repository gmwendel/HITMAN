# *HITMAN* 2.0

Neural likelihood-ratio event reconstruction for optical neutrino detectors with
arbitrary geometry — rebuilt in **JAX**. Networks are [equinox](https://github.com/patrick-kidger/equinox)
pytrees, the learned log likelihood-to-evidence ratio is a differentiable JAX function,
and inference (multistart MLE, NUTS) runs as jitted programs that can be exported as a
single compiled graph (e.g. via jax2tf for C++ deployment).

Method: extended maximum likelihood with per-hit (**hitnet**) and total-charge
(**chargenet**) neural ratio estimators — see the Reference below. The 1.x
TensorFlow implementation lives on the `main` branch; this branch (`v2.0-jax`)
preserves its feature definitions and training semantics for model parity.

## What changed vs 1.x

- **No hypothesis copies:** hits live in a flat array with an `event_id` index;
  per-event terms compose with `segment_sum`, and HitNet's first layer is evaluated
  separably (`embed_hit` + `embed_hyp`), so during reconstruction the per-hit work is
  done once per event instead of once per likelihood call.
- **Logit-native networks:** trained with BCE-on-logits; the network output *is*
  log r (the post-training sigmoid→linear re-save hack is gone).
- **Inference in the graph:** `multistart_mle` (seed → top-k → vmapped projected-Adam
  descent in a scaled parameter space, replacing hand-tuned per-parameter learning
  rates) and `sample_nuts` (blackjax, window adaptation) are pure jittable JAX.
- **SBI hygiene:** optional balanced-NRE regularizer (arXiv:2208.13624) in the loss;
  classifier reliability/ECE, ratio self-normalization, and SBC rank/coverage
  diagnostics in `hitman.diagnostics`.

## Install

```bash
conda create -n hitman_jax python=3.12
conda activate hitman_jax
pip install -e .          # CPU jax; install a CUDA jaxlib first for GPU training
```

## Usage sketch

```python
import jax, numpy as np
from hitman.data import RatDSExtractor
from hitman.nn import HitNet, ChargeNet
from hitman.train import fit
from hitman.likelihood import make_event_nll
from hitman.inference import multistart_mle, cylinder_seeds, sample_nuts
from hitman.inference.nuts import make_event_logdensity, box_log_prior

batch = RatDSExtractor(["sim_0.ntuple.root", "sim_1.ntuple.root"]).load()
key = jax.random.PRNGKey(0)

# train (hitnet: per-hit rows; chargenet: per-event rows)
hitnet = fit(HitNet(key=key), np.asarray(batch.hits),
             np.asarray(batch.hyp)[np.asarray(batch.event_id)], key=key).model
chargenet = fit(ChargeNet(key=key), np.asarray(batch.charge),
                np.asarray(batch.hyp), key=key).model

# reconstruct one event
dev = batch.to_device()
mask = np.asarray(dev.event_id) == 0
nll = make_event_nll(hitnet, chargenet, dev.hits[mask], dev.charge[0])
seeds = cylinder_seeds(key, 2000, radius=900., half_height=900.,
                       t_range=(-5, 5), e_range=(0.5, 3.0))
lo, hi = ...  # detector box in (x,y,z,zen,az,t,E)
mle = multistart_mle(nll, seeds, bounds=(lo, hi))

# posterior for the same event
logdensity = make_event_logdensity(nll, box_log_prior(lo, hi))
posterior = sample_nuts(logdensity, mle.theta, key)
```

Tests: `python -m pytest tests` (includes an integration test that runs if an Eos
simulation ntuple is available).

## Reference

[FreeDOM](https://github.com/philippeller/freeDOM/) reconstruction using the same technique but focused on IceCube analysis.

This method was published as:

A flexible event reconstruction based on machine learning and likelihood principles

Philipp Eller (Munich, Tech. U.), Aaron T. Fienberg (Penn State U.), Jan Weldert (Mainz U., Inst. Phys.), Garrett Wendel (Penn State U.), Sebastian Böser (Mainz U., Inst. Phys.) et al.

e-Print: [2208.10166](https://arxiv.org/abs/2208.10166)
DOI: [10.1016/j.nima.2023.168011](https://doi.org/10.1016/j.nima.2023.168011) (publication)
Nucl.Instrum.Meth.A 1048 (2023), 168011

Please cite as
```
@article{Eller:2022xvi,
    author = {Eller, Philipp and Fienberg, Aaron T. and Weldert, Jan and Wendel, Garrett and B\"oser, Sebastian and Cowen, D. F.},
    title = "{A flexible event reconstruction based on machine learning and likelihood principles}",
    eprint = "2208.10166",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    doi = "10.1016/j.nima.2023.168011",
    journal = "Nucl. Instrum. Meth. A",
    volume = "1048",
    pages = "168011",
    year = "2023"
}
```
