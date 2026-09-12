import jax
import jax.numpy as jnp
import numpy as np

from hitman import EventBatch, event_log_ratio
from hitman.wc.data.sampling import time_shuffle
from hitman.nn import ChargeNet, HitNet


def test_time_shuffle_moves_absolute_times_but_preserves_dt():
    hits = jnp.array([[100.0, 0.0, 0.0, 5.0], [0.0, 100.0, 0.0, 7.0]])
    batch = EventBatch(
        hyp=jnp.array([[0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 1.5]]),
        charge=jnp.array([[2.0, 2.0]]),
        hits=hits,
        event_id=jnp.array([0, 0], dtype=jnp.int32),
    )
    shuffled = time_shuffle(jax.random.PRNGKey(3), batch, sigma=50.0)

    shift = float(shuffled.hyp[0, 5] - batch.hyp[0, 5])
    assert abs(shift) > 1.0  # the absolute time really moved
    np.testing.assert_allclose(
        np.asarray(shuffled.hits[:, 3] - batch.hits[:, 3]), shift, rtol=1e-5
    )

    # hitnet features depend on dt only -> log-ratio unchanged by the shift
    hitnet = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(1))
    before = event_log_ratio(hitnet, chargenet, batch, batch.hyp)
    after = event_log_ratio(hitnet, chargenet, shuffled, shuffled.hyp)
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-4)
