import unittest
import tensorflow as tf
import numpy as np

# We assume the new layers will be in hitman.neural_nets.d2h_layers
from hitman.neural_nets.d2h_layers import D2hSymmetrizedLayer4Param, D2hSymmetrizedLayer8Param

class TestD2hSymmetrizedLayers(unittest.TestCase):
    def setUp(self):
        # Basic test inputs
        self.batch_size = 2
        self.num_sensors = 4
        self.pitch = 10.0
        self.detector_scale = 100.0
        
        # Scat, Abs
        self.hyp_4param = tf.constant([
            [0.5, 5000.0],
            [0.8, 8000.0]
        ], dtype=tf.float32)
        
        # PMT Coordinates:
        # PMT 1: On +X wall
        # PMT 2: On -Y wall
        # PMT 3: On +Z wall
        # PMT 4: On -X wall
        self.pmt_positions = tf.constant([[
            [100.0, 10.0, 20.0],
            [15.0, -100.0, -5.0],
            [-10.0, 30.0, 100.0],
            [-100.0, 20.0, -10.0]
        ]], dtype=tf.float32)
        # Tile to match batch size
        self.pmt_positions = tf.tile(self.pmt_positions, [self.batch_size, 1, 1])

        self.layer_4 = D2hSymmetrizedLayer4Param(pitch=self.pitch, detector_scale=self.detector_scale)
        self.layer_8 = D2hSymmetrizedLayer8Param(pitch=self.pitch, detector_scale=self.detector_scale)

    def test_layer_instantiation(self):
        self.assertIsNotNone(self.layer_4)
        self.assertIsNotNone(self.layer_8)
        
    def test_canonical_eigenbasis(self):
        # We can extract the _compute_eigenbasis static or internal method to test
        # Let's test it directly on the layer by mocking or checking internal tensors
        # if we factor it out as a @tf.function or method.
        # Let's define the expected inward normals for our test PMTs:
        # PMT 1 (+X): [-1, 0, 0]
        # PMT 2 (-Y): [0, 1, 0]
        # PMT 3 (+Z): [0, 0, -1]
        # PMT 4 (-X): [1, 0, 0]
        expected_u = tf.constant([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ], dtype=tf.float32)
        
        u_i, e_L, e_T1, e_T2 = self.layer_4._compute_eigenbasis(self.pmt_positions)
        
        # Check inward normals
        np.testing.assert_allclose(u_i[0].numpy(), expected_u.numpy(), atol=1e-5)
        
        # Check Longitudinal axis (absolute of inward normal)
        expected_e_L = tf.abs(expected_u)
        np.testing.assert_allclose(e_L[0].numpy(), expected_e_L.numpy(), atol=1e-5)
        
        # Check Transverse 1 (Z-axis, k = [0, 0, 1])
        expected_e_T1 = tf.constant([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=tf.float32)
        np.testing.assert_allclose(e_T1[0].numpy(), expected_e_T1.numpy(), atol=1e-5)
        
        # Check Transverse 2 (k x e_L)
        # e_T2 = cross([0,0,1], e_L)
        # For PMT 1 (e_L = [1,0,0]): cross([0,0,1], [1,0,0]) = [0, 1, 0]
        # For PMT 2 (e_L = [0,1,0]): cross([0,0,1], [0,1,0]) = [-1, 0, 0]
        # For PMT 3 (e_L = [0,0,1]): cross([0,0,1], [0,0,1]) = [0, 0, 0]
        # For PMT 4 (e_L = [1,0,0]): cross([0,0,1], [1,0,0]) = [0, 1, 0]
        expected_e_T2 = tf.constant([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=tf.float32)
        np.testing.assert_allclose(e_T2[0].numpy(), expected_e_T2.numpy(), atol=1e-5)

    def test_quadrant_folding(self):
        # We will extract _compute_quadrant_folding internally to test
        # Let's mock the eigenbasis for PMT 1 (+X face: e_T1=[0,0,1], e_T2=[0,1,0])
        # Position: [100, -10, -20], scale: 100
        # Expected: r_norm = [1.0, -0.1, -0.2]
        # Projection into T1 (Z): -0.2 -> abs = 0.2
        # Projection into T2 (Y): -0.1 -> abs = 0.1
        pmt_pos = tf.constant([[[100.0, -10.0, -20.0]]], dtype=tf.float32)
        e_T1 = tf.constant([[[0.0, 0.0, 1.0]]], dtype=tf.float32)
        e_T2 = tf.constant([[[0.0, 1.0, 0.0]]], dtype=tf.float32)
        
        abs_T1, abs_T2 = self.layer_4._compute_quadrant_folding(pmt_pos, e_T1, e_T2)
        
        np.testing.assert_allclose(abs_T1[0].numpy(), [[0.2]], atol=1e-5)
        np.testing.assert_allclose(abs_T2[0].numpy(), [[0.1]], atol=1e-5)

    def test_relative_diffusion_kinematics(self):
        # Given a sensor and a vertex, test Z_rel, n_T1, n_T2, rho
        pmt_pos = tf.constant([[[100.0, 10.0, 20.0]]], dtype=tf.float32)
        vertex = tf.constant([[50.0, 5.0, -5.0]], dtype=tf.float32) # (Batch, 3)
        # Delta r (Sensor - Vertex): [50.0, 5.0, 25.0]
        
        u_i = tf.constant([[[-1.0, 0.0, 0.0]]], dtype=tf.float32) # Inward normal for +X wall
        e_T1 = tf.constant([[[0.0, 0.0, 1.0]]], dtype=tf.float32) # Z axis
        e_T2 = tf.constant([[[0.0, 1.0, 0.0]]], dtype=tf.float32) # Y axis
        
        # Z_rel = Delta r dot u_i = [50, 5, 25] dot [-1, 0, 0] = -50.0
        # n_T1 = Delta r dot e_T1 = 25.0
        # n_T2 = Delta r dot e_T2 = 5.0
        # rho = sqrt(25^2 + 5^2) = sqrt(625 + 25) = sqrt(650) = 25.495097...
        
        Z_rel, n_T1, n_T2, rho = self.layer_8._compute_relative_kinematics(pmt_pos, vertex, u_i, e_T1, e_T2)
        
        np.testing.assert_allclose(Z_rel[0].numpy(), [[-50.0]], atol=1e-5)
        np.testing.assert_allclose(n_T1[0].numpy(), [[25.0]], atol=1e-5)
        np.testing.assert_allclose(n_T2[0].numpy(), [[5.0]], atol=1e-5)
        np.testing.assert_allclose(rho[0].numpy(), [[25.4950975679]], atol=1e-5)

    def test_transverse_directional_symmetrization(self):
        # We need to test the sgn*(x) logic restoring parity relative to the quadrant.
        # obs_r: [100.0, -10.0, -20.0]
        pmt_pos = tf.constant([[[100.0, -10.0, -20.0]]], dtype=tf.float32)
        e_T1 = tf.constant([[[0.0, 0.0, 1.0]]], dtype=tf.float32) # Z axis
        e_T2 = tf.constant([[[0.0, 1.0, 0.0]]], dtype=tf.float32) # Y axis
        
        n_T1 = tf.constant([[[25.0]]], dtype=tf.float32) # Delta r dot e_T1
        n_T2 = tf.constant([[[5.0]]], dtype=tf.float32)  # Delta r dot e_T2
        rho = tf.constant([[[25.4950975679]]], dtype=tf.float32)
        
        # Symmetrization logic:
        # sign_T1 = sgn(obs_r dot e_T1) = sgn(-20.0) = -1.0
        # sign_T2 = sgn(obs_r dot e_T2) = sgn(-10.0) = -1.0
        # dir_T1 = (n_T1 * sign_T1) / rho = (25.0 * -1.0) / 25.495 = -0.98058
        # dir_T2 = (n_T2 * sign_T2) / rho = (5.0 * -1.0) / 25.495 = -0.196116
        
        dir_T1, dir_T2 = self.layer_4._compute_transverse_symmetrization(pmt_pos, n_T1, n_T2, rho, e_T1, e_T2)
        
        np.testing.assert_allclose(dir_T1[0].numpy(), [[-0.9805806]], atol=1e-5)
        np.testing.assert_allclose(dir_T2[0].numpy(), [[-0.1961161]], atol=1e-5)
        
    def test_medium_opacities(self):
        # hyp is [Scat, Abs] = [0.5, 5000.0]
        # pitch = 10.0
        # op_abs = 10.0 / 5000.0 = 0.002
        # op_scat = 10.0 / 0.5 = 20.0
        hyp = tf.constant([[0.5, 5000.0]], dtype=tf.float32)
        
        op_abs, op_scat = self.layer_4._compute_medium_opacities(hyp)
        
        np.testing.assert_allclose(op_abs.numpy(), [[0.002]], atol=1e-5)
        np.testing.assert_allclose(op_scat.numpy(), [[20.0]], atol=1e-5)

    def test_call_and_gradient_stability(self):
        # We test the full call method and verify that gradients flow smoothly
        # without NaN even when delta_r has zero components (e.g. perfectly on-axis)
        
        # 1. Test 8-Parameter (Vertex provided)
        # We will make one PMT exactly aligned with the vertex on the T1/T2 axes to test the rho singularity
        pmt_pos_8 = tf.constant([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]], dtype=tf.float32)
        # Vertex is exactly on the X axis, so for PMT 1 (+X wall), n_T1=0 and n_T2=0
        vertex = tf.Variable([[50.0, 0.0, 0.0]], dtype=tf.float32)
        hyp_8 = tf.Variable([[0.5, 5000.0]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # For the 8-param layer, the input expects [hyp, pmt_pos, vertex]
            # Actually, we should match the 8-param input spec:
            # "Sensor locations, scintillator properties, and the 4 vertex parameters"
            # Let's say vertex_input is [X, Y, Z, T]. We only need X, Y, Z for the spatial math.
            vertex_4d = tf.concat([vertex, tf.zeros((1, 1))], axis=-1) # Add T=0
            out_8 = self.layer_8([hyp_8, pmt_pos_8, vertex_4d])
            loss_8 = tf.reduce_sum(out_8)
            
        grads_8 = tape.gradient(loss_8, [hyp_8, vertex])
        
        # Verify no NaNs
        self.assertFalse(np.any(np.isnan(grads_8[0].numpy())), "NaN in hyp gradients (8-param)")
        self.assertFalse(np.any(np.isnan(grads_8[1].numpy())), "NaN in vertex gradients (8-param)")
        # Ensure gradients are actually flowing (not all exactly zero for the vertex)
        self.assertFalse(np.all(grads_8[1].numpy() == 0), "Zero gradients for vertex (8-param)")

        # 2. Test 4-Parameter (No Vertex)
        pmt_pos_4 = tf.constant([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]], dtype=tf.float32)
        hyp_4 = tf.Variable([[0.5, 5000.0]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            out_4 = self.layer_4([hyp_4, pmt_pos_4])
            loss_4 = tf.reduce_sum(out_4)
            
        grads_4 = tape.gradient(loss_4, [hyp_4])
        self.assertFalse(np.any(np.isnan(grads_4[0].numpy())), "NaN in hyp gradients (4-param)")
        self.assertFalse(np.all(grads_4[0].numpy() == 0), "Zero gradients for hyp (4-param)")

    def test_shapenet_integration(self):
        # Verify that we can build a Keras model using the D2h layer without shape mismatches
        hyp_in = tf.keras.Input(shape=(2,), name="hyp_in")
        obs_in = tf.keras.Input(shape=(None, 3), name="obs_in")
        total_hits_in = tf.keras.Input(shape=(1,), name="total_hits_in")
        
        # 4-param integration
        d2h_layer = D2hSymmetrizedLayer4Param(pitch=10.0, detector_scale=100.0)
        x = d2h_layer([hyp_in, obs_in])
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        outputs = tf.squeeze(outputs, axis=-1)
        pmf = tf.keras.layers.Softmax(axis=1)(outputs)
        expected_hits = pmf * total_hits_in
        
        model = tf.keras.Model(inputs=[hyp_in, obs_in, total_hits_in], outputs=expected_hits)
        
        # Test forward pass with dynamic batch and sensors
        dummy_hyp = tf.ones((2, 2))
        dummy_obs = tf.ones((2, 5, 3))
        dummy_hits = tf.ones((2, 1))
        
        out = model([dummy_hyp, dummy_obs, dummy_hits])
        self.assertEqual(out.shape, (2, 5))

if __name__ == '__main__':
    unittest.main()
