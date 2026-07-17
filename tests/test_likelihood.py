import jax
import jax.numpy as jnp
import numpy as np

from hitman import EventBatch, event_log_ratio
from hitman.likelihood import make_event_nll
from hitman.nn import ChargeNet, HitNet


def _toy_batch():
    # 3 events with 2, 0, 4 hits — includes an empty event.
    hits = jnp.array(
        [
            [100.0, 0.0, 0.0, 5.0],
            [0.0, 100.0, 0.0, 7.0],
            [0.0, 0.0, 100.0, 3.0],
            [50.0, 50.0, 0.0, 4.0],
            [0.0, 50.0, 50.0, 6.0],
            [50.0, 0.0, 50.0, 8.0],
        ]
    )
    event_id = jnp.array([0, 0, 2, 2, 2, 2], dtype=jnp.int32)
    hyp = jnp.array(
        [
            [10.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.5],
            [0.0, 10.0, 0.0, 1.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 10.0, 1.5, 3.0, 0.0, 2.5],
        ]
    )
    charge = jnp.array([[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]])
    return EventBatch(hyp=hyp, charge=charge, hits=hits, event_id=event_id)


def _nets():
    return (
        HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
        ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(1)),
    )


def test_event_log_ratio_matches_naive_loop():
    hitnet, chargenet = _nets()
    batch = _toy_batch()
    result = np.asarray(event_log_ratio(hitnet, chargenet, batch, batch.hyp))

    for e in range(batch.n_events):
        mask = np.asarray(batch.event_id) == e
        expected = sum(
            float(hitnet(batch.hits[i], batch.hyp[e])) for i in np.where(mask)[0]
        ) + float(chargenet(batch.charge[e], batch.hyp[e]))
        np.testing.assert_allclose(result[e], expected, rtol=1e-5)


def test_make_event_nll_consistent_with_batch_path():
    hitnet, chargenet = _nets()
    batch = _toy_batch()
    e = 2
    mask = np.asarray(batch.event_id) == e
    nll = make_event_nll(hitnet, chargenet, batch.hits[mask], batch.charge[e])
    batch_value = float(event_log_ratio(hitnet, chargenet, batch, batch.hyp)[e])
    np.testing.assert_allclose(float(nll(batch.hyp[e])), -batch_value, rtol=2e-5)


def test_empty_event_gets_charge_term_only():
    hitnet, chargenet = _nets()
    batch = _toy_batch()
    result = event_log_ratio(hitnet, chargenet, batch, batch.hyp)
    expected = float(chargenet(batch.charge[1], batch.hyp[1]))
    np.testing.assert_allclose(float(result[1]), expected, rtol=1e-5)
