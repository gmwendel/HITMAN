import tensorflow as tf
import numpy as np

def get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=32768, shuffle=True, split=None, val_fraction=0.1):
    """
    Constructs a tf.data.Dataset that yields ((pmt_positions, hypotheses), charges)
    for Poisson ChargeNet training. Uses a fast batched Python generator mapped to 
    vectorized Tensor operations to prevent GPU OOM crashes on array initialization.
    """
    N_events = charges.shape[0]
    N_sensors = pmt_positions.shape[0]
    
    events_per_batch = max(1, batch_size // N_sensors)
    
    pmt_tensor = tf.constant(pmt_positions, dtype=tf.float32)
    
    start_idx = 0
    end_idx = N_events
    if split is not None:
        val_size = max(1, int(N_events * val_fraction))
        if split == 'val':
            end_idx = val_size
        elif split == 'train':
            start_idx = val_size
            
    def batch_generator():
        indices = np.arange(start_idx, end_idx)
        if shuffle:
            np.random.shuffle(indices)
        
        # Yield pre-sliced batches of events from Python to avoid GIL overhead on millions of singles
        for i in range(0, len(indices), events_per_batch):
            batch_indices = indices[i:i+events_per_batch]
            yield charges[batch_indices], charge_hyp[batch_indices]
            
    ds = tf.data.Dataset.from_generator(
        batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, N_sensors), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    )
    
    def map_batch(batch_charges, batch_hyp):
        current_events = tf.shape(batch_charges)[0]
        
        batch_pmt = tf.tile(tf.expand_dims(pmt_tensor, 0), [current_events, 1, 1])
        batch_hyp_rep = tf.tile(tf.expand_dims(batch_hyp, 1), [1, N_sensors, 1])
        
        flat_pmt = tf.reshape(batch_pmt, [-1, 3])
        flat_hyp = tf.reshape(batch_hyp_rep, [-1, 3])
        flat_charges = tf.reshape(batch_charges, [-1])
        
        return (flat_pmt, flat_hyp), flat_charges

    ds = ds.map(map_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    
    return ds