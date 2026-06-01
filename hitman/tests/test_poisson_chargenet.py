import pytest
import numpy as np
import tensorflow as tf
from hitman.neural_nets.poisson_chargenet import get_poisson_chargenet, poisson_nll_loss

def test_poisson_chargenet_architecture():
    # Provide dummy normalization matrices
    hyp_norm = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    obs_norm = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    # Instantiate the network
    model = get_poisson_chargenet(hyp_norm=hyp_norm, obs_norm=obs_norm)
    
    # 3 PMT pos inputs + 3 parameter inputs
    assert model.inputs[0].shape == (None, 3)
    assert model.inputs[1].shape == (None, 3)

    # Test forward pass
    dummy_pmt = tf.zeros((1, 3))
    dummy_params = tf.zeros((1, 3))
    
    # Output natively predicts lambda via exp, so it must be strictly positive
    out_lambda = model([dummy_pmt, dummy_params])
    assert out_lambda.shape == (1, 1)
    assert np.all(out_lambda.numpy() > 0), "Exp output must be positive to represent a valid Poisson rate"

def test_poisson_nll_loss():
    # Test the custom loss calculation predicting log-rate (z): e^z - k * z
    
    # If lambda = e (approx 2.718), then z = ln(lambda) = 1.0
    # Observed k = 1.0
    # NLL = e^1 - 1.0 * 1.0 = e - 1.0
    
    z_pred = tf.constant([1.0], dtype=tf.float32)
    y_true = tf.constant([1.0], dtype=tf.float32)
    
    loss = poisson_nll_loss(y_true, z_pred)
    assert np.isclose(loss.numpy(), np.e - 1.0, atol=1e-4)
    
    # Test epsilon protection for log(0) - equivalent to extremely negative z
    z_pred_extreme = tf.constant([-100.0], dtype=tf.float32)
    y_true_zero = tf.constant([1.0], dtype=tf.float32)
    loss_extreme = poisson_nll_loss(y_true_zero, z_pred_extreme)
    # Should not be NaN
    assert not np.isnan(loss_extreme.numpy())
