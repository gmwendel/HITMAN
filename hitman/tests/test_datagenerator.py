import numpy as np
import pytest
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
