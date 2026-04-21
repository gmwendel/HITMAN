import tensorflow as tf
import numpy as np

def get_shape_dataset(shape_targets, charge_hyp, pmt_positions, vertex, batch_size=1024, shuffle=True, split=None, val_fraction=0.1):
    N_events = shape_targets.shape[0]
    N_sensors = pmt_positions.shape[0]
    
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
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield shape_targets[batch_indices], charge_hyp[batch_indices], vertex[batch_indices]
            
    ds = tf.data.Dataset.from_generator(
        batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, N_sensors), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    )
    
    def map_batch(batch_shapes, batch_hyp, batch_vertex):
        current_events = tf.shape(batch_shapes)[0]
        batch_pmt = tf.tile(tf.expand_dims(pmt_tensor, 0), [current_events, 1, 1])
        batch_shapes_64 = tf.cast(batch_shapes, tf.float64)
        return (batch_hyp, batch_pmt, batch_vertex), batch_shapes_64

    ds = ds.map(map_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    
    return ds

def get_acceptance_dataset(rate_targets, charge_hyp, vertex, batch_size=1024, shuffle=True, split=None, val_fraction=0.1):
    N_events = rate_targets.shape[0]
    
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
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield charge_hyp[batch_indices], vertex[batch_indices], rate_targets[batch_indices]
            
    ds = tf.data.Dataset.from_generator(
        batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    )
    
    def map_batch(batch_hyp, batch_vertex, batch_rates):
        return (batch_hyp, batch_vertex), batch_rates

    ds = ds.map(map_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    
    return ds
