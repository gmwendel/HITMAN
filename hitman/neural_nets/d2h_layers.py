import tensorflow as tf

class D2hSymmetrizedLayer4Param(tf.keras.layers.Layer):
    """
    4-Parameter Transformation Layer for D2h Symmetrized Latent Space.
    Inputs: 
        - Sensor locations (X, Y, Z)
        - Scintillator properties (Scat, Abs)
        (No vertex provided, implicitly assumes origin or relative inputs are handled elsewhere,
         but according to the spec, 4-param means NO vertex provided).
         
    Note: Sensors must be offset from the bounding cube edges/corners to avoid 
    L_infinity boundary ambiguities during dynamic normal computation.
    """
    def __init__(self, pitch, detector_scale, **kwargs):
        super(D2hSymmetrizedLayer4Param, self).__init__(**kwargs)
        self.pitch = pitch
        self.detector_scale = detector_scale

    def _compute_eigenbasis(self, obs_r):
        """
        Constructs the Canonical Eigenbasis from the absolute sensor positions.
        obs_r: Tensor of shape (Batch, N_sensors, 3) representing PMT coordinates.
        """
        # 1. Find the axis of maximum absolute value (the mounted wall)
        abs_r = tf.abs(obs_r)
        d_idx = tf.argmax(abs_r, axis=-1)  # Shape: (Batch, N_sensors)
        
        # 2. Create the standard basis vector e_d
        # tf.one_hot converts the axis index into a unit vector [1,0,0], [0,1,0], or [0,0,1]
        e_d = tf.one_hot(d_idx, depth=3, dtype=tf.float32) # Shape: (Batch, N_sensors, 3)
        
        # 3. Extract the sign of the coordinate on the mounted wall
        # We use stop_gradient to prevent the optimizer from trying to flow through the sign function
        sign_d = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_d, axis=-1, keepdims=True)))
        
        # 4. Inward-facing normal vector (u_i)
        # If sign is positive (on the +X wall), inward normal is -X. So u_i = -sign_d * e_d
        u_i = -sign_d * e_d
        
        # 5. Abstract Fiber Eigenbasis
        e_L = tf.abs(u_i) # Longitudinal unsigned axis
        
        # Stacking axis (Conventionally Z = [0, 0, 1])
        k_hat = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        # Broadcast k_hat to match batch and sensor dims
        e_T1 = tf.tile(tf.reshape(k_hat, (1, 1, 3)), [tf.shape(obs_r)[0], tf.shape(obs_r)[1], 1])
        
        # In-Plane Transverse Axis (e_T2 = k x e_L)
        e_T2 = tf.linalg.cross(e_T1, e_L)
        
        return u_i, e_L, e_T1, e_T2

    def _compute_quadrant_folding(self, obs_r, e_T1, e_T2):
        """
        Global Boundary Projections (Quadrant Folding).
        Reduces the absolute spatial context to a 2D boundary proximity map.
        """
        r_norm = obs_r / self.detector_scale
        
        abs_T1 = tf.abs(tf.reduce_sum(r_norm * e_T1, axis=-1, keepdims=True))
        abs_T2 = tf.abs(tf.reduce_sum(r_norm * e_T2, axis=-1, keepdims=True))
        
        return abs_T1, abs_T2

    def _compute_relative_kinematics(self, obs_r, vertex, u_i, e_T1, e_T2):
        """
        Extracts fundamental kinematics relative to the vertex in the local sensor frame.
        obs_r: (Batch, Sensors, 3)
        vertex: (Batch, 3)
        """
        v_exp = tf.expand_dims(vertex, axis=1)
        delta_r = obs_r - v_exp
        
        Z_rel = tf.reduce_sum(delta_r * u_i, axis=-1, keepdims=True)
        n_T1 = tf.reduce_sum(delta_r * e_T1, axis=-1, keepdims=True)
        n_T2 = tf.reduce_sum(delta_r * e_T2, axis=-1, keepdims=True)
        
        # Absolute transverse impact parameter with machine-epsilon stabilizer to prevent NaN gradients
        rho = tf.sqrt(tf.square(n_T1) + tf.square(n_T2) + 1e-7)
        
        return Z_rel, n_T1, n_T2, rho

    def _compute_transverse_symmetrization(self, obs_r, n_T1, n_T2, rho, e_T1, e_T2):
        """
        Transverse Directional Symmetrization.
        Restores orientation relative to the quadrant-folded boundaries.
        """
        # sign*(x) where sign*(0) = 1
        sign_T1 = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_T1, axis=-1, keepdims=True) + 1e-12))
        sign_T2 = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_T2, axis=-1, keepdims=True) + 1e-12))
        
        dir_T1 = (n_T1 * sign_T1) / rho
        dir_T2 = (n_T2 * sign_T2) / rho
        
        return dir_T1, dir_T2

    def _compute_medium_opacities(self, hyp):
        """
        Maps macroscopic cross-sections to dimensionless interactions per lattice pitch.
        hyp: (Batch, 2) -> [Scat, Abs]
        """
        # hyp[:, 0] is Scat, hyp[:, 1] is Abs
        # We need to expand dims so it can be broadcasted to (Batch, 1, 1) or returned as (Batch, 1)
        scat = tf.expand_dims(hyp[:, 0], axis=1)
        abs_len = tf.expand_dims(hyp[:, 1], axis=1)
        
        op_abs = self.pitch / abs_len
        op_scat = self.pitch / scat
        
        return op_abs, op_scat

    def call(self, inputs):
        # 4-Parameter Inputs: [hyp, obs_r]
        # Since this assumes no vertex is provided, we implicitly use the origin for kinematics.
        hyp, obs_r = inputs
        
        u_i, e_L, e_T1, e_T2 = self._compute_eigenbasis(obs_r)
        abs_T1, abs_T2 = self._compute_quadrant_folding(obs_r, e_T1, e_T2)
        
        # Origin vertex for 4-param model
        batch_size = tf.shape(obs_r)[0]
        origin = tf.zeros((batch_size, 3), dtype=tf.float32)
        
        Z_rel, n_T1, n_T2, rho = self._compute_relative_kinematics(obs_r, origin, u_i, e_T1, e_T2)
        dir_T1, dir_T2 = self._compute_transverse_symmetrization(obs_r, n_T1, n_T2, rho, e_T1, e_T2)
        
        op_abs, op_scat = self._compute_medium_opacities(hyp)
        # Expand dims to match (Batch, Sensors, 1) for concatenation
        num_sensors = tf.shape(obs_r)[1]
        op_abs = tf.tile(tf.expand_dims(op_abs, 1), [1, num_sensors, 1])
        op_scat = tf.tile(tf.expand_dims(op_scat, 1), [1, num_sensors, 1])
        
        # Scale spatial inputs by detector_scale
        Z_rel_norm = Z_rel / self.detector_scale
        rho_norm = rho / self.detector_scale
        
        # Concatenate final 8-dimensional tensor
        final_tensor = tf.concat([
            Z_rel_norm,
            rho_norm,
            dir_T1,
            dir_T2,
            abs_T1,
            abs_T2,
            op_abs,
            op_scat
        ], axis=-1)
        
        return final_tensor

class D2hSymmetrizedLayer8Param(tf.keras.layers.Layer):
    """
    8-Parameter Transformation Layer for D2h Symmetrized Latent Space.
    Inputs:
        - Sensor locations (X, Y, Z)
        - Scintillator properties (Scat, Abs)
        - Vertex parameters (X, Y, Z, T)
        
    Note: Sensors must be offset from the bounding cube edges/corners to avoid 
    L_infinity boundary ambiguities during dynamic normal computation.
    """
    def __init__(self, pitch, detector_scale, **kwargs):
        super(D2hSymmetrizedLayer8Param, self).__init__(**kwargs)
        self.pitch = pitch
        self.detector_scale = detector_scale

    def _compute_eigenbasis(self, obs_r):
        """
        Constructs the Canonical Eigenbasis from the absolute sensor positions.
        obs_r: Tensor of shape (Batch, N_sensors, 3) representing PMT coordinates.
        """
        abs_r = tf.abs(obs_r)
        d_idx = tf.argmax(abs_r, axis=-1)  
        e_d = tf.one_hot(d_idx, depth=3, dtype=tf.float32) 
        sign_d = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_d, axis=-1, keepdims=True)))
        u_i = -sign_d * e_d
        e_L = tf.abs(u_i) 
        k_hat = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        e_T1 = tf.tile(tf.reshape(k_hat, (1, 1, 3)), [tf.shape(obs_r)[0], tf.shape(obs_r)[1], 1])
        e_T2 = tf.linalg.cross(e_T1, e_L)
        return u_i, e_L, e_T1, e_T2

    def _compute_quadrant_folding(self, obs_r, e_T1, e_T2):
        """
        Global Boundary Projections (Quadrant Folding).
        Reduces the absolute spatial context to a 2D boundary proximity map.
        """
        r_norm = obs_r / self.detector_scale
        
        abs_T1 = tf.abs(tf.reduce_sum(r_norm * e_T1, axis=-1, keepdims=True))
        abs_T2 = tf.abs(tf.reduce_sum(r_norm * e_T2, axis=-1, keepdims=True))
        
        return abs_T1, abs_T2

    def _compute_relative_kinematics(self, obs_r, vertex, u_i, e_T1, e_T2):
        """
        Extracts fundamental kinematics relative to the vertex in the local sensor frame.
        obs_r: (Batch, Sensors, 3)
        vertex: (Batch, 3)
        """
        v_exp = tf.expand_dims(vertex, axis=1)
        delta_r = obs_r - v_exp
        
        Z_rel = tf.reduce_sum(delta_r * u_i, axis=-1, keepdims=True)
        n_T1 = tf.reduce_sum(delta_r * e_T1, axis=-1, keepdims=True)
        n_T2 = tf.reduce_sum(delta_r * e_T2, axis=-1, keepdims=True)
        
        # Absolute transverse impact parameter with machine-epsilon stabilizer to prevent NaN gradients
        rho = tf.sqrt(tf.square(n_T1) + tf.square(n_T2) + 1e-7)
        
        return Z_rel, n_T1, n_T2, rho

    def _compute_transverse_symmetrization(self, obs_r, n_T1, n_T2, rho, e_T1, e_T2):
        """
        Transverse Directional Symmetrization.
        Restores orientation relative to the quadrant-folded boundaries.
        """
        # sign*(x) where sign*(0) = 1
        sign_T1 = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_T1, axis=-1, keepdims=True) + 1e-12))
        sign_T2 = tf.stop_gradient(tf.sign(tf.reduce_sum(obs_r * e_T2, axis=-1, keepdims=True) + 1e-12))
        
        dir_T1 = (n_T1 * sign_T1) / rho
        dir_T2 = (n_T2 * sign_T2) / rho
        
        return dir_T1, dir_T2

    def _compute_medium_opacities(self, hyp):
        """
        Maps macroscopic cross-sections to dimensionless interactions per lattice pitch.
        hyp: (Batch, 2) -> [Scat, Abs]
        """
        scat = tf.expand_dims(hyp[:, 0], axis=1)
        abs_len = tf.expand_dims(hyp[:, 1], axis=1)
        
        op_abs = self.pitch / abs_len
        op_scat = self.pitch / scat
        
        return op_abs, op_scat

    def call(self, inputs):
        # 8-Parameter Inputs: [hyp, obs_r, vertex]
        hyp, obs_r, vertex_input = inputs
        
        # vertex_input is (Batch, 4) -> [X, Y, Z, T]
        # We only need X, Y, Z for the spatial math
        vertex = vertex_input[:, :3]
        
        u_i, e_L, e_T1, e_T2 = self._compute_eigenbasis(obs_r)
        abs_T1, abs_T2 = self._compute_quadrant_folding(obs_r, e_T1, e_T2)
        
        Z_rel, n_T1, n_T2, rho = self._compute_relative_kinematics(obs_r, vertex, u_i, e_T1, e_T2)
        dir_T1, dir_T2 = self._compute_transverse_symmetrization(obs_r, n_T1, n_T2, rho, e_T1, e_T2)
        
        op_abs, op_scat = self._compute_medium_opacities(hyp)
        # Expand dims to match (Batch, Sensors, 1) for concatenation
        num_sensors = tf.shape(obs_r)[1]
        op_abs = tf.tile(tf.expand_dims(op_abs, 1), [1, num_sensors, 1])
        op_scat = tf.tile(tf.expand_dims(op_scat, 1), [1, num_sensors, 1])
        
        # Scale spatial inputs by detector_scale
        Z_rel_norm = Z_rel / self.detector_scale
        rho_norm = rho / self.detector_scale
        
        # Concatenate final 8-dimensional tensor
        final_tensor = tf.concat([
            Z_rel_norm,
            rho_norm,
            dir_T1,
            dir_T2,
            abs_T1,
            abs_T2,
            op_abs,
            op_scat
        ], axis=-1)
        
        return final_tensor
