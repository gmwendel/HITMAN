import tensorflow as tf
import numpy as np
import pytest
from hitman.neural_nets.factorized_chargenet import get_shape_net, get_acceptance_net, factorized_chargenet_trafo

def test_factorized_networks():
    hyp_norm = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    obs_norm = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    
    shape_net = get_shape_net(layers=2, nodes=128, hyp_norm=hyp_norm, obs_norm=obs_norm)
    acc_net = get_acceptance_net(layers=2, nodes=128, hyp_norm=hyp_norm, obs_norm=obs_norm)
    
    # 2 events, 3 hypothesis params (E, scat, abs)
    test_hyp = tf.constant([[0.5, 0.6, 8500.0], [0.2, 0.8, 5000.0]], dtype=tf.float32)
    # 2 events, 64 sensors, 3 coordinates each
    test_obs = tf.random.uniform((2, 64, 3), dtype=tf.float32)
    
    # Test ShapeNet
    shape_pred = shape_net([test_hyp, test_obs])
    assert shape_pred.shape == (2, 64)
    # Softmax PMF should sum to 1.0 for each event
    sums = tf.reduce_sum(shape_pred, axis=1)
    assert np.allclose(sums.numpy(), np.array([1.0, 1.0]))
    
    # Test AcceptanceNet
    # AcceptanceNet predicts global array acceptance, so it only depends on the material properties,
    # and maybe vertex position. But not individual PMT positions.
    # The architecture should be designed so AcceptanceNet only takes hypothesis.
    acc_pred_correct = acc_net([test_hyp])
    assert acc_pred_correct.shape == (2, 1)
    
    # Since we use Softplus or ReLU, output must be non-negative
    assert tf.reduce_all(acc_pred_correct >= 0.0).numpy()
