"""Round-5 export gate: the distilled search student + teacher primitives, and the full
compiled solver graph with each candidate activation, must survive StableHLO
serialization and round-trip bit-for-bit (design decision #1 -- the cppflow/StableHLO
deployment path must support every op the cheaper activation emits).

A cheaper search activation that fails lowering is dead regardless of speed, so this
runs the same `jax.export` round-trip as scripts/export_compiled_mle_stablehlo.py over:
  1. the TEACHER penalized value+grad primitive (mish),
  2. each STUDENT penalized value+grad primitive (mish/swish/softplus/relu/hardswish),
  3. the whole make_compiled_mle while_loop/top_k/solve graph built on a student net of
     each activation (proves the activation lowers inside the fused optimizer graph too).

Usage: export_student_teacher_stablehlo.py [act1,act2,...]   (default: all five)
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      "/tank/playground/hitman-sbi-modernization/.jax_cache")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from jax import export
from hitman.data import RatDSExtractor
from hitman.nn import HitNet, ChargeNet
from hitman.nn.mlp import ACTIVATIONS
from hitman.wc.inference.batched import pad_events
from hitman.wc.inference.compiled import make_compiled_mle, CompiledMLEConfig
from hitman.wc.inference.seq import _build_primitives

ROOT = "/tank/playground/hitman-sbi-modernization"
RUN = f"{ROOT}/training_runs/run10_recipe_5M"
STUD = f"{ROOT}/training_runs/r5_students"
DATA = f"{ROOT}/datagen/testpoints/data"
N_PAD = 160
acts = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ACTIVATIONS)

teacher = eqx.tree_deserialise_leaves(f"{RUN}/hitnet.eqx", HitNet(key=jax.random.PRNGKey(0)))
chargenet = eqx.tree_deserialise_leaves(f"{RUN}/chargenet.eqx", ChargeNet(key=jax.random.PRNGKey(1)))
cfg = CompiledMLEConfig(e_range=(0.5, 10.0))

b = RatDSExtractor([f"{DATA}/e3MeV_zen000.root"]).load()
counts = np.asarray(b.charge[:, 1]).astype(int); keep = np.where(counts <= N_PAD)[0][:5]
p = pad_events(b, indices=keep, n_pad=N_PAD)
hits, pid, t, mask, charge = p.hits, p.pmt_id, p.t, p.mask, p.charge
e_teacher = np.asarray(jax.vmap(teacher.embed_hit)(hits[0]))


def roundtrip_prim(name, fn, args):
    exp = export.export(jax.jit(fn))(*args)
    blob = exp.serialize(); re = export.deserialize(blob)
    d = jax.jit(fn)(*args); s = re.call(*args)
    ok = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
             for x, y in zip(jax.tree_util.tree_leaves(d), jax.tree_util.tree_leaves(s)))
    print(f"  [{name}] {len(blob)} bytes  max|Δ| direct-vs-serialized = {ok:.2e}")
    return ok


print("=== primitive value+grad round-trip (penalized NLL) ===")
pt = _build_primitives(teacher, chargenet, cfg, N_PAD)
vg_t = pt["vg_pen"]
a_t = (jnp.asarray(p_ := np.zeros(8)), jnp.asarray(e_teacher), jnp.asarray(mask[0]), jnp.asarray(charge[0]))
worst = roundtrip_prim("teacher mish vg_pen", vg_t, a_t)

for act in acts:
    st = eqx.tree_deserialise_leaves(
        f"{STUD}/student_{act}_d3.eqx",
        HitNet(width=32, depth=3, key=jax.random.PRNGKey(0), activation=act))
    ps = _build_primitives(st, chargenet, cfg, N_PAD)
    e_s = np.asarray(jax.vmap(st.embed_hit)(hits[0]))
    a_s = (jnp.asarray(np.zeros(8)), jnp.asarray(e_s), jnp.asarray(mask[0]), jnp.asarray(charge[0]))
    worst = max(worst, roundtrip_prim(f"student {act} vg_pen", ps["vg_pen"], a_s))

print("=== full compiled-solver graph round-trip, per activation ===")
for act in acts:
    st = eqx.tree_deserialise_leaves(
        f"{STUD}/student_{act}_d3.eqx",
        HitNet(width=32, depth=3, key=jax.random.PRNGKey(0), activation=act))
    fn = make_compiled_mle(st, chargenet, cfg, n_pad=N_PAD)
    args = (hits[0], pid[0], t[0], mask[0], charge[0])
    exp = export.export(fn)(*args); blob = exp.serialize(); re = export.deserialize(blob)
    md = 0.0
    for i in range(5):
        ai = (hits[i], pid[i], t[i], mask[i], charge[i])
        md = max(md, float(np.max(np.abs(np.asarray(fn(*ai)) - np.asarray(re.call(*ai))))))
    print(f"  [compiled {act}] {len(blob)} bytes  max|Δ| = {md:.2e}")
    worst = max(worst, md)

print(f"\nEXPORT GATE: worst round-trip max|Δ| across all graphs = {worst:.2e}")
print("PASS" if worst < 1e-3 else "FAIL")
