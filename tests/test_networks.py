import jax
import jax.numpy as jnp
import numpy as np

from hitman.nn import ChargeNet, HitNet


def test_hitnet_split_embedding_equals_direct_forward():
    key = jax.random.PRNGKey(0)
    hitnet = HitNet(width=64, depth=3, key=key)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    hits = jax.random.normal(k1, (50, 4)) * jnp.array([500.0, 500.0, 500.0, 20.0])
    hyp = jnp.array([100.0, -50.0, 200.0, 1.1, 3.0, 2.0, 1.8])

    direct = jax.vmap(lambda h: hitnet(h, hyp))(hits)
    pre = jax.vmap(hitnet.embed_hit)(hits) + hitnet.embed_hyp(hyp)
    split = jax.vmap(hitnet.logit_from_embedding)(pre)

    np.testing.assert_allclose(np.asarray(split), np.asarray(direct), rtol=2e-5, atol=2e-5)


def test_networks_output_scalar_logits():
    hitnet = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(1))
    hit = jnp.array([100.0, 0.0, -100.0, 12.0])
    charge = jnp.array([40.0, 40.0])
    hyp = jnp.zeros(7).at[6].set(1.0)
    assert hitnet(hit, hyp).shape == ()
    assert chargenet(charge, hyp).shape == ()


def test_hitnet_gradients_flow_to_hypothesis():
    hitnet = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    hit = jnp.array([100.0, 0.0, -100.0, 12.0])
    hyp = jnp.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0])
    grad = jax.grad(lambda p: hitnet(hit, p))(hyp)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.any(np.asarray(grad) != 0.0)
