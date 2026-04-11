import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch
from hitman.tools.datagenerator import DataGenerator

def test_indom_shuffling_is_cached():
    # Create dummy data representing PMT hits
    N_hits = 1000
    N_params = 6
    # pmtX, pmtY, pmtZ, time, ...
    x = np.random.uniform(size=(N_hits, 5)).astype(np.float32)
    # hypotheses
    t = np.random.uniform(size=(N_hits, N_params)).astype(np.float32)
    
    with patch('numpy.unique', wraps=np.unique) as mock_unique:
        # Initialize generator (which triggers the first inDOM shuffle)
        gen = DataGenerator(x, t, batch_size=32, shuffle='inDOM')
        
        # Ensure np.unique was actually used during initialization
        initial_call_count = mock_unique.call_count
        assert initial_call_count > 0, "np.unique should be called during init for inDOM shuffling"
        
        # Reset mock to monitor epoch boundaries
        mock_unique.reset_mock()
        
        # Trigger an epoch end, which currently recalculates the sorting structures
        gen.on_epoch_end()
        
        # If the index grouping structures are properly cached, np.unique (a massive CPU bottleneck)
        # should never be called again after initialization.
        assert mock_unique.call_count == 0, f"np.unique was called {mock_unique.call_count} times during on_epoch_end, meaning structures are NOT cached!"

def test_indom_disk_caching():
    N_hits = 1000
    N_params = 6
    x = np.random.uniform(size=(N_hits, 5)).astype(np.float32)
    t = np.random.uniform(size=(N_hits, N_params)).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "indom_cache.npz")
        
        # First initialization should calculate and write to cache
        gen1 = DataGenerator(x, t, batch_size=32, shuffle='inDOM', cache_file=cache_path)
        assert os.path.exists(cache_path), "Cache file was not created!"
        
        # Second initialization should read from cache and completely skip np.unique
        with patch('numpy.unique', wraps=np.unique) as mock_unique:
            gen2 = DataGenerator(x, t, batch_size=32, shuffle='inDOM', cache_file=cache_path)
            assert mock_unique.call_count == 0, "np.unique was called despite cache existing!"
            
            # Verify data integrity
            np.testing.assert_array_equal(gen1.sort_idx, gen2.sort_idx)
            np.testing.assert_array_equal(gen1.split_idx, gen2.split_idx)

