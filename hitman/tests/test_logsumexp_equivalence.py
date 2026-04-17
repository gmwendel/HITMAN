import unittest
import tensorflow as tf
import numpy as np

class TestLogSumExpEquivalence(unittest.TestCase):
    def test_mathematical_equivalence(self):
        # We will create some mock raw logits for 5 sensors and 2 events
        # We will use float64 to ensure high precision in comparison
        logits = tf.constant([
            [1.5, 2.0, -1.0, 5.0, 0.0],
            [10.0, -20.0, 5.0, 8.0, 2.0]
        ], dtype=tf.float64)

        # Mock observed hit counts for the sensors
        hits = tf.constant([
            [2.0, 5.0, 0.0, 10.0, 1.0],
            [100.0, 0.0, 10.0, 50.0, 5.0]
        ], dtype=tf.float64)

        # -------------------------------------------------------------
        # 1. OLD METHOD: Softmax + Poisson
        # -------------------------------------------------------------
        # In the old method, we exponentiated the logits (Softmax)
        pmf_old = tf.nn.softmax(logits, axis=1)
        
        # We scaled by total hits
        total_hits = tf.reduce_sum(hits, axis=1, keepdims=True)
        expected_hits = pmf_old * total_hits
        
        # And computed the Poisson loss
        # Poisson Loss = expected_hits - hits * ln(expected_hits)
        poisson_loss_per_event = tf.reduce_sum(expected_hits - hits * tf.math.log(expected_hits + 1e-9), axis=1)
        old_loss_mean = tf.reduce_mean(poisson_loss_per_event)

        # -------------------------------------------------------------
        # 2. NEW METHOD: Multinomial Cross-Entropy using LogSoftmax
        # -------------------------------------------------------------
        # We manually compute -sum(hits * log_softmax(logits)) to avoid any Keras internal target normalization
        # when hits don't sum to 1.
        def multinomial_crossentropy(y_true, y_pred):
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.nn.log_softmax(y_pred), axis=1))
            
        new_loss_mean = multinomial_crossentropy(hits, logits)

        # -------------------------------------------------------------
        # 3. VERIFY EQUIVALENCE (UP TO A CONSTANT)
        # -------------------------------------------------------------
        # To prove they are equivalent, we will show that their gradients with respect to the logits are identical.
        # This proves the neural network will learn exactly the same weights.
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(logits)
            
            # Old forward pass
            pmf = tf.nn.softmax(logits, axis=1)
            e_hits = pmf * total_hits
            loss_old = tf.reduce_mean(tf.reduce_sum(e_hits - hits * tf.math.log(e_hits + 1e-9), axis=1))
            
            # New forward pass
            loss_new = multinomial_crossentropy(hits, logits)
            
        grad_old = tape.gradient(loss_old, logits)
        grad_new = tape.gradient(loss_new, logits)
        
        # The gradients should be exactly the same!
        np.testing.assert_allclose(grad_old.numpy(), grad_new.numpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
