import jax
import numpy as np

from hitman.nn import HitNet
from hitman.train import DeviceData, exact_polish, hit_batch
from tests.test_resident import _toy_data


def test_exact_polish_deterministic_and_improves():
    batch, pmt_pos = _toy_data(n_events=300, seed=21)
    data = DeviceData.from_batch(batch, pmt_pos)
    net = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    m1, h1 = exact_polish(net, data, hit_batch, data.n_hits, key=jax.random.PRNGKey(3),
                          max_iters=5, chunk_size=256, n_pairings=2, verbose=False)
    m2, h2 = exact_polish(net, data, hit_batch, data.n_hits, key=jax.random.PRNGKey(3),
                          max_iters=5, chunk_size=256, n_pairings=2, verbose=False)
    # frozen objective + deterministic optimizer => bitwise-reproducible trajectory
    assert h1 == h2
    # exact loss decreases monotonically-ish over L-BFGS iterations
    losses = [x[0] for x in h1]
    assert losses[-1] < losses[0]
    assert np.isfinite(losses).all()
