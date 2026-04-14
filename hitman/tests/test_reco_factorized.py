import tensorflow as tf
import numpy as np
import pytest
from hitman.reco_joint_factorized import newton_raphson_epsilon, tfLLH_joint_factorized

def test_newton_raphson_epsilon():
    # Create some mock data
    # Let's say we have 2 events and 3 sensors
    # S_ij = Y * mu * f_ij
    # S has shape (N_hyp, N_events, N_sensors)
    # Lambda_sig has shape (N_hyp, N_events)
    # b has shape (N_sensors,)
    
    # We test for N_hyp=1
    S = tf.constant([[[10.0, 20.0, 30.0],
                      [5.0,  15.0, 25.0]]], dtype=tf.float32)
                      
    Lambda_sig = tf.constant([[60.0, 45.0]], dtype=tf.float32)
    b = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)
    charges = tf.constant([[5.0, 10.0, 15.0],
                           [2.0, 5.0, 8.0]], dtype=tf.float32)
    
    # Run the Newton-Raphson optimizer to find epsilon
    # We should expect it to converge cleanly
    eps = newton_raphson_epsilon(S, Lambda_sig, b, charges, max_iter=20, tol=1e-5)
    
    # The derivative should be near 0 at this epsilon
    # dL/deps = -sum(Lambda_sig) + sum(k * S / (eps * S + b))
    # For eps:
    eps_val = eps[0, 0]
    
    term1 = -tf.reduce_sum(Lambda_sig[0])
    term2 = tf.reduce_sum(charges * S[0] / (eps_val * S[0] + b))
    derivative = term1 + term2
    
    # The derivative should be very close to zero if we converged
    assert tf.abs(derivative) < 1e-4

def test_tfLLH_joint_factorized_runs():
    # Simple check that the LLH function doesn't crash with dummy inputs
    # Need to properly mock ShapeNet and AcceptanceNet
    class MockShapeNet(tf.keras.Model):
        def call(self, inputs):
            hyp, obs = inputs
            # Return uniform PMF
            return tf.ones((tf.shape(hyp)[0], tf.shape(obs)[1]), dtype=tf.float32) / tf.cast(tf.shape(obs)[1], tf.float32)
            
    class MockAccNet(tf.keras.Model):
        def call(self, inputs):
            # Return constant acceptance
            return tf.ones((tf.shape(inputs)[0], 1), dtype=tf.float32) * 0.5
            
    shape_net = MockShapeNet()
    acc_net = MockAccNet()
    
    # N_events=2, N_sensors=3
    charges = np.array([[10, 20, 30], [5, 15, 25]], dtype=np.float32)
    pmt_positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    hyp_batch = np.array([[2.0, 3.0]], dtype=np.float32)
    
    # Y = yield = 100000.0 * Energy. Let's mock a yield vector for the 2 events
    yields = tf.constant([10000.0, 20000.0], dtype=tf.float32)
    
    # Run LLH
    llhs = tfLLH_joint_factorized(charges, pmt_positions, hyp_batch, yields, shape_net, acc_net)
    
    assert llhs.shape == (1,)
    assert not np.isnan(llhs.numpy()[0])
