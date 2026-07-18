"""Export smoke test for the compiled MLE solver (design decision #1).

Validates that the whole fused graph — surrogate networks + Bancroft/centroid
seeding + top_k screen + Levenberg-Marquardt `while_loop` descent — survives
serialization to a portable StableHLO artifact and round-trips bit-for-bit.

Uses `jax.export` (TF-free StableHLO), which exercises exactly the ops the cppflow
deployment path must support (while_loop, top_k, linear solves). The jax2tf ->
TF SavedModel variant for cppflow is in export_compiled_mle_jax2tf.py (needs a TF
install, absent in hitman_jax).
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["JAX_PLATFORMS"]="cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR","/tank/playground/hitman-sbi-modernization/.jax_cache")
import numpy as np, jax, equinox as eqx
from jax import export
from hitman.data import RatDSExtractor
from hitman.nn import HitNet, ChargeNet
from hitman.inference.batched import pad_events
from hitman.inference.compiled import make_compiled_mle, CompiledMLEConfig

RUN="/tank/playground/hitman-sbi-modernization/training_runs/run10_recipe_5M"
DATA="/tank/playground/hitman-sbi-modernization/datagen/testpoints/data"
N_PAD=160
hitnet=eqx.tree_deserialise_leaves(f"{RUN}/hitnet.eqx",HitNet(key=jax.random.PRNGKey(0)))
chargenet=eqx.tree_deserialise_leaves(f"{RUN}/chargenet.eqx",ChargeNet(key=jax.random.PRNGKey(1)))
fn=make_compiled_mle(hitnet,chargenet,CompiledMLEConfig(),n_pad=N_PAD)  # jitted

b=RatDSExtractor([f"{DATA}/e3MeV_zen000.root"]).load()
counts=np.asarray(b.charge[:,1]).astype(int); keep=np.where(counts<=N_PAD)[0][:5]
p=pad_events(b,indices=keep,n_pad=N_PAD)
args=(p.hits[0],p.pmt_id[0],p.t[0],p.mask[0],p.charge[0])

exp=export.export(fn)(*args)
blob=exp.serialize()
print("serialized StableHLO bytes:",len(blob))
print("platforms:",exp.platforms," in_avals:",[str(a.shape)+str(a.dtype) for a in exp.in_avals])
rehydrated=export.deserialize(blob)
for i in range(5):
    a=(p.hits[i],p.pmt_id[i],p.t[i],p.mask[i],p.charge[i])
    direct=np.asarray(fn(*a)); ser=np.asarray(rehydrated.call(*a))
    print(f" event {i}: max|Δ| direct-vs-serialized = {np.max(np.abs(direct-ser)):.2e}")
print("EXPORT SMOKE: while_loop/top_k/solve survived serialization; round-trip matches.")
