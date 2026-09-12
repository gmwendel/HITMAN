"""jax2tf -> TF SavedModel export of the compiled MLE solver (cppflow deploy path).

This is design-decision-#1 option (a): package the fused solver as a TF SavedModel
so the existing EOS::HitmanProc cppflow loader runs it with ONE call per event
(replacing the NLopt x NLLH loop). READY TO RUN once TensorFlow is available
(absent in the hitman_jax env: `pip list | grep -i tensor` -> empty). The
equivalent TF-free StableHLO round-trip is validated in export_smoke.py (bit-exact).

Run:
    pip install tensorflow            # into hitman_jax (or a TF-only sidecar env)
    JAX_PLATFORMS=cpu python export_compiled_mle_jax2tf.py

`native_serialization=True` emits StableHLO inside the SavedModel (the same IR the
smoke test round-tripped), so the while_loop / top_k / linear-solve ops the compiled
optimizer emits are carried through as XLA custom-calls rather than lowered to TF
ops — verify the target TF/XLA build supports them at load time.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import numpy as np
import equinox as eqx
import tensorflow as tf                         # noqa: E402  (absent in hitman_jax)
from jax.experimental import jax2tf             # noqa: E402

from hitman.nn import HitNet, ChargeNet
from hitman.wc.inference.compiled import make_compiled_mle, CompiledMLEConfig

RUN = "/tank/playground/hitman-sbi-modernization/training_runs/run10_recipe_5M"
OUT = "/tank/playground/hitman-sbi-modernization/training_runs/run10_recipe_5M/compiled_mle_savedmodel"
N_PAD = 160

hitnet = eqx.tree_deserialise_leaves(f"{RUN}/hitnet.eqx", HitNet(key=jax.random.PRNGKey(0)))
chargenet = eqx.tree_deserialise_leaves(f"{RUN}/chargenet.eqx", ChargeNet(key=jax.random.PRNGKey(1)))
solver = make_compiled_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=N_PAD)

tf_fn = jax2tf.convert(solver, native_serialization=True)
module = tf.Module()
module.solve = tf.function(tf_fn, autograph=False, input_signature=[
    tf.TensorSpec([N_PAD, 4], tf.float32, name="hits"),
    tf.TensorSpec([N_PAD], tf.int32, name="pmt_id"),
    tf.TensorSpec([N_PAD], tf.float32, name="t"),
    tf.TensorSpec([N_PAD], tf.float32, name="mask"),
    tf.TensorSpec([2], tf.float32, name="charge"),
])
tf.saved_model.save(module, OUT)
print("wrote", OUT)

# parity check vs JAX
reloaded = tf.saved_model.load(OUT)
h = np.zeros((N_PAD, 4), np.float32); h[:20, :3] = np.random.randn(20, 3) * 500; h[:20, 3] = 5.0
m = np.zeros(N_PAD, np.float32); m[:20] = 1.0
args = (h, np.zeros(N_PAD, np.int32), h[:, 3], m, np.array([26.0, 20.0], np.float32))
jax_out = np.asarray(solver(*args))
tf_out = reloaded.solve(*[tf.constant(a) for a in args]).numpy()
print("max|Δ| jax-vs-savedmodel:", float(np.max(np.abs(jax_out - tf_out))))  # expect < 1e-5
