import jax

# On GPU, this JAX version's default matmul precision is TF32 (10-bit mantissa,
# rel err ~5e-4) — fine for training throughput, wrong for equivalence/parity
# assertions. Pin the suite to true float32 so tests mean the same thing on any
# backend; production code may opt into TF32 explicitly as a speed lever.
jax.config.update("jax_default_matmul_precision", "highest")
