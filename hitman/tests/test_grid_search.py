import pytest
import numpy as np
import tensorflow as tf
from hitman.hitman_reco import iterative_random_search

def test_iterative_random_search():
    # Create mock models that always return 0 for LLH calculation
    # We just want to test the mathematical looping and bounds shrinking logic
    
    # Mock HitNet: input is [(N*num_params, 5), (N*num_params, 3)], output is (N*num_params, 1)
    class MockHitNet(tf.keras.Model):
        def call(self, inputs):
            h, p = inputs
            return tf.zeros((tf.shape(h)[0], 1))

    # Mock ChargeNet: input is [(N, 2), (N, 3)], output is (N, 1)
    class MockChargeNet(tf.keras.Model):
        def call(self, inputs):
            c, p = inputs
            return tf.zeros((tf.shape(p)[0], 1))
            
    hitnet = MockHitNet()
    chargenet = MockChargeNet()
    
    event = {
        'hits': np.zeros((10, 5), dtype=np.float32),
        'total_charge': np.zeros((2,), dtype=np.float32)
    }
    
    samples_per_stage = 100
    stages = 3
    zoom_factor = 0.5
    vram_batch_size = 50
    
    best_llh, best_point = iterative_random_search(
        hitnet=hitnet, 
        chargenet=chargenet, 
        event=event, 
        samples_per_stage=samples_per_stage, 
        stages=stages, 
        zoom_factor=zoom_factor, 
        vram_batch_size=vram_batch_size
    )
    
    # Since our mock networks always return 0 for NLLH, best_llh should be exactly 0
    assert best_llh == 0.0
    
    # The best point must lie within the absolute bounds
    assert 0.1 <= best_point[0] <= 0.6
    assert 0.2 <= best_point[1] <= 1.0
    assert 2000.0 <= best_point[2] <= 15000.0
