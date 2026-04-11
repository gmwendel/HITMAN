import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=2 ** 12, shuffle='free', time_spread=50):
        assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."

        self.batch_size = int(batch_size / 2)  # half true labels half false labels
        self.data = np.array(x)
        self.params = np.array(t)

        self._labels = np.concatenate([
            np.ones((self.batch_size, 1), dtype=self.data.dtype),
            np.zeros((self.batch_size, 1), dtype=self.data.dtype)
        ])

        # spread absolute time values (for hitnet)
        if len(self.data[0]) > 4 and len(self.params[0]) > 5:
            time_shifts = np.random.normal(0, time_spread, len(self.data))
            self.data[:, 3] += time_shifts
            self.params[:, 5] += time_shifts

        if shuffle == 'inDOM':
            pmt_coords = self.data[:, 0:3]
            unique_coords, inverse_indices = np.unique(pmt_coords, axis=0, return_inverse=True)
            
            self.sort_idx = np.argsort(inverse_indices)
            sorted_params = self.params[self.sort_idx]
            sorted_inv = inverse_indices[self.sort_idx]
            
            _, counts = np.unique(sorted_inv, return_counts=True)
            self.split_idx = np.cumsum(counts)[:-1]
            
            self.grouped_params = np.split(sorted_params, self.split_idx)
            
            self.unsort_idx = np.empty_like(self.sort_idx)
            self.unsort_idx[self.sort_idx] = np.arange(len(self.sort_idx))
            
            self.shuffle_params_inDOM()
        else:
            self.shuffled_params = []

        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def get_norms(self):

        return hyp_norm, obs_norm

    def shuffle_params_inDOM(self):
        shuffled_grouped_params = [np.random.permutation(g) for g in self.grouped_params]
        shuffled_sorted_params = np.concatenate(shuffled_grouped_params)
        
        self.shuffled_params = shuffled_sorted_params[self.unsort_idx]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes)  # mix between batches
        if len(self.shuffled_params) > 0:
            self.shuffle_params_inDOM()

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        x = np.take(self.data, indexes_temp, axis=0)
        t = np.take(self.params, indexes_temp, axis=0)
        if len(self.shuffled_params) == 0:
            tr = np.random.permutation(t)
        else:
            tr = np.take(self.shuffled_params, indexes_temp, axis=0)

        d_X = np.concatenate([x, x], axis=0)
        d_T = np.concatenate([t, tr], axis=0)
        
        # Dynamically generate labels to match current batch size
        cur_batch = len(x)
        d_labels = np.concatenate([
            np.ones((cur_batch, 1), dtype=d_X.dtype),
            np.zeros((cur_batch, 1), dtype=d_X.dtype)
        ])

        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return (d_X, d_T), d_labels

    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))

def get_hitnet_dataset(hits, hit_hyp, batch_size=4096, shuffle='inDOM', time_spread=50, split=None, val_fraction=0.1):
    """
    Constructs a tf.data.Dataset for HitNet training.
    Replaces the massive Python Sequence loops with a fast Python generator 
    that yields pre-batched True and False samples directly to a tf.data pipeline.
    """
    N_hits = hits.shape[0]
    N_params = hit_hyp.shape[1]
    
    half_batch = batch_size // 2
    
    # Pre-calculate inDOM structures exactly once if requested
    if shuffle == 'inDOM':
        pmt_coords = hits[:, 0:3]
        unique_coords, inverse_indices = np.unique(pmt_coords, axis=0, return_inverse=True)
        
        sort_idx = np.argsort(inverse_indices)
        sorted_params = hit_hyp[sort_idx]
        sorted_inv = inverse_indices[sort_idx]
        
        _, counts = np.unique(sorted_inv, return_counts=True)
        split_idx = np.cumsum(counts)[:-1]
        
        grouped_params = np.split(sorted_params, split_idx)
        
        unsort_idx = np.empty_like(sort_idx)
        unsort_idx[sort_idx] = np.arange(len(sort_idx))
    
    # Apply time spread mapping (using TF to avoid modifying original array)
    def map_time_spread(batch_x, batch_t, batch_y):
        if time_spread > 0 and batch_x.shape[1] > 4 and batch_t.shape[1] > 5:
            current_batch_size = tf.shape(batch_x)[0]
            time_shifts = tf.random.normal([current_batch_size], 0, time_spread, dtype=tf.float32)
            
            # Create a sparse update tensor for the time column (index 3)
            indices_x = tf.stack([tf.range(current_batch_size, dtype=tf.int32), tf.fill([current_batch_size], 3)], axis=1)
            updates_x = time_shifts
            batch_x = tf.tensor_scatter_nd_add(batch_x, indices_x, updates_x)
            
            # Create a sparse update tensor for the param time column (index 5)
            indices_t = tf.stack([tf.range(current_batch_size, dtype=tf.int32), tf.fill([current_batch_size], 5)], axis=1)
            updates_t = time_shifts
            batch_t = tf.tensor_scatter_nd_add(batch_t, indices_t, updates_t)
            
        return (batch_x, batch_t), batch_y

    def batch_generator():
        # Handle splitting boundaries
        start_idx = 0
        end_idx = N_hits
        if split is not None:
            val_size = max(1, int(N_hits * val_fraction))
            if split == 'val':
                end_idx = val_size
            elif split == 'train':
                start_idx = val_size
        
        indices = np.arange(start_idx, end_idx)
        np.random.shuffle(indices)
        
        # Perform the inDOM shuffle for the entire epoch
        if shuffle == 'inDOM':
            shuffled_grouped_params = [np.random.permutation(g) for g in grouped_params]
            shuffled_sorted_params = np.concatenate(shuffled_grouped_params)
            shuffled_hit_hyp = shuffled_sorted_params[unsort_idx]
        else:
            shuffled_hit_hyp = hit_hyp[np.random.permutation(len(hit_hyp))]
        
        # Yield pre-computed full batches to prevent GIL locking on millions of single yields
        for i in range(0, len(indices), half_batch):
            batch_indices = indices[i:i+half_batch]
            current_half = len(batch_indices)
            
            # True samples
            batch_x_true = hits[batch_indices]
            batch_t_true = hit_hyp[batch_indices]
            
            # False samples (using the pre-shuffled hypotheses)
            batch_x_false = hits[batch_indices]
            batch_t_false = shuffled_hit_hyp[batch_indices]
            
            batch_x = np.concatenate([batch_x_true, batch_x_false], axis=0)
            batch_t = np.concatenate([batch_t_true, batch_t_false], axis=0)
            batch_y = np.concatenate([np.ones((current_half, 1), dtype=np.float32), np.zeros((current_half, 1), dtype=np.float32)], axis=0)
            
            # Mix True and False samples uniformly
            mix_p = np.random.permutation(current_half * 2)
            yield batch_x[mix_p], batch_t[mix_p], batch_y[mix_p]
            
    ds = tf.data.Dataset.from_generator(
        batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, hits.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(None, N_params), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )
    
    ds = ds.map(map_time_spread, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    
    return ds
