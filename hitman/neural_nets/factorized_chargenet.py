import tensorflow as tf
import numpy as np

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))

from hitman.neural_nets.d2h_layers import D2hSymmetrizedLayer4Param, D2hSymmetrizedLayer8Param

def get_shape_net(activation=mish, layers=2, nodes=128, hyp_norm=None, acc_hyp_norm=None, obs_norm=None, use_vertex=False, use_d2h=True):
    hyp_input = tf.keras.Input(shape=(2,), name="shape_hyp_in")
    obs_input = tf.keras.Input(shape=(None, 3), name="shape_obs_in")
    
    # We define the detector constants based on the training dataset.
    # Currently, pitch = 10.0mm (typical for LiquidO WbLS grid), and scale = 1000.0mm
    # Note: Sensors must be offset from bounding box edges/corners.
    pitch = 10.0
    detector_scale = 1000.0

    if use_d2h:
        # 1. Expand features through trainable physics prior
        features = FiberPhysicsLayer(init_L_fiber=1000.0)(hyp_input)
        
        # 2. Normalize features
        norm_features = tf.keras.layers.Normalization(mean=acc_hyp_norm[1][:5], variance=acc_hyp_norm[0][:5]**2, axis=-1)(features)
        
        if use_vertex:
            vertex_input = tf.keras.Input(shape=(3,), name="shape_vertex_in")
            d2h_tensor = D2hSymmetrizedLayer8Param(pitch=pitch, detector_scale=detector_scale)(
                [norm_features, obs_input, vertex_input]
            )
            inputs_list = [hyp_input, obs_input, vertex_input]
        else:
            d2h_tensor = D2hSymmetrizedLayer4Param(pitch=pitch, detector_scale=detector_scale)(
                [norm_features, obs_input]
            )
            inputs_list = [hyp_input, obs_input]
        x = d2h_tensor
    else:
        # OLD BASELINE ARCHITECTURE (Cartesian + R)
        inputs_list = [hyp_input, obs_input]
        num_sensors = tf.shape(obs_input)[1]
        
        R = tf.sqrt(tf.reduce_sum(tf.square(obs_input), axis=-1, keepdims=True))
        R_scaled = R / 100.0 
        
        norm_hyp = tf.keras.layers.Normalization(mean=hyp_norm[1], variance=hyp_norm[0]**2, axis=-1)(hyp_input)
        norm_obs = tf.keras.layers.Normalization(mean=obs_norm[1], variance=obs_norm[0]**2, axis=-1)(obs_input)
        
        h_tiled = tf.tile(tf.expand_dims(norm_hyp, 1), [1, num_sensors, 1])
        x = tf.concat([h_tiled, norm_obs, R_scaled], axis=-1)

    for i in range(layers):
        x = tf.keras.layers.Dense(nodes, activation=activation)(x)

    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    outputs = tf.squeeze(outputs, axis=-1)
    
    # Cast logits to float64 for mixed-precision LogSumExp stability
    logits_fp64 = tf.cast(outputs, tf.float64)
    
    return tf.keras.Model(inputs=inputs_list, outputs=logits_fp64, name="ShapeNet")

class FiberPhysicsLayer(tf.keras.layers.Layer):
    def __init__(self, init_L_fiber=1000.0, **kwargs):
        super(FiberPhysicsLayer, self).__init__(**kwargs)
        self.init_L_fiber = init_L_fiber

    def build(self, input_shape):
        # Trainable log-scale parameter to keep L_fiber strictly positive.
        # Initialize w_fiber = 0 so that L_fiber = init_L_fiber.
        self.w_fiber = self.add_weight(
            name="w_fiber",
            shape=(),
            initializer="zeros",
            trainable=True
        )
        super(FiberPhysicsLayer, self).build(input_shape)

    def call(self, inputs):
        scat = inputs[:, 0:1]
        abs_len = inputs[:, 1:2]
        
        L_fiber = self.init_L_fiber * tf.exp(self.w_fiber)
        
        # 1 / L_eff = 1 / L_abs + 1 / L_fiber
        inv_L_eff = (1.0 / (abs_len + 1e-12)) + (1.0 / L_fiber)
        L_eff = 1.0 / inv_L_eff
        
        L_D = tf.sqrt((L_eff * scat) / 3.0)
        
        # omega = Sigma_scat / (Sigma_scat + Sigma_abs_total)
        Sigma_scat = 1.0 / (scat + 1e-12)
        omega = Sigma_scat / (Sigma_scat + inv_L_eff)
        
        log_scat = tf.math.log(scat + 1e-12)
        log_abs = tf.math.log(abs_len + 1e-12)
        log_L_eff = tf.math.log(L_eff + 1e-12)
        log_LD = tf.math.log(L_D + 1e-12)
        
        return tf.concat([log_scat, log_abs, log_L_eff, log_LD, omega], axis=-1)

    def get_config(self):
        config = super(FiberPhysicsLayer, self).get_config()
        config.update({"init_L_fiber": self.init_L_fiber})
        return config

def get_acceptance_net(activation="silu", layers=2, nodes=128, hyp_norm=None, obs_norm=None):
    hyp_input = tf.keras.Input(shape=(2,), name="acc_hyp_in")
    vertex_input = tf.keras.Input(shape=(3,), name="acc_vertex_in")
    
    # 1. Expand features through trainable physics prior
    features = FiberPhysicsLayer(init_L_fiber=1000.0)(hyp_input)
    
    # Concatenate the physics priors with the 3D vertex
    combined_features = tf.concat([features, vertex_input], axis=-1)
    
    # 2. We apply a static Normalization layer initialized with the global dataset statistics 
    # calculated in the training script to preserve the 1-to-1 deterministic physical mapping
    # Note: Because the feature dim increased to 8, hyp_norm MUST be of shape (2, 8)
    h = tf.keras.layers.Normalization(mean=hyp_norm[1], variance=hyp_norm[0]**2, axis=-1)(combined_features)
    
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation)(h)
        
    outputs = tf.keras.layers.Dense(1, activation="linear", name="acc_dense_out", 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4),
                                    bias_initializer=tf.keras.initializers.Constant(-3.5))(h)
    return tf.keras.Model(inputs=[hyp_input, vertex_input], outputs=outputs, name="AcceptanceNet")

def get_wrapped_acceptance_net(acc_net):
    """
    Temporary wrapper to provide backwards compatibility with old plotting scripts.
    It takes the log-acceptance output (z_eps) and exponentiates it to return the 
    linear geometric acceptance fraction (mu_geom).
    """
    hyp_input = tf.keras.Input(shape=(2,), name="acc_hyp_in_wrapped")
    vertex_input = tf.keras.Input(shape=(3,), name="acc_vertex_in_wrapped")
    z_eps = acc_net([hyp_input, vertex_input])
    mu_geom = tf.exp(z_eps)
    return tf.keras.Model(inputs=[hyp_input, vertex_input], outputs=mu_geom, name="AcceptanceNet_Wrapped")
