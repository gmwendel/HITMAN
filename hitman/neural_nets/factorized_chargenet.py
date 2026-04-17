import tensorflow as tf
import numpy as np

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))

from hitman.neural_nets.d2h_layers import D2hSymmetrizedLayer4Param, D2hSymmetrizedLayer8Param

def get_shape_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None, use_vertex=False, use_d2h=True):
    hyp_input = tf.keras.Input(shape=(2,), name="shape_hyp_in")
    obs_input = tf.keras.Input(shape=(None, 3), name="shape_obs_in")
    
    # We define the detector constants based on the training dataset.
    # Currently, pitch = 10.0mm (typical for LiquidO WbLS grid), and scale = 1000.0mm
    # Note: Sensors must be offset from bounding box edges/corners.
    pitch = 10.0
    detector_scale = 1000.0

    if use_d2h:
        if use_vertex:
            vertex_input = tf.keras.Input(shape=(4,), name="shape_vertex_in")
            d2h_tensor = D2hSymmetrizedLayer8Param(pitch=pitch, detector_scale=detector_scale)(
                [hyp_input, obs_input, vertex_input]
            )
            inputs_list = [hyp_input, obs_input, vertex_input]
        else:
            d2h_tensor = D2hSymmetrizedLayer4Param(pitch=pitch, detector_scale=detector_scale)(
                [hyp_input, obs_input]
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

def get_acceptance_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None):
    hyp_input = tf.keras.Input(shape=(2,), name="acc_hyp_in")
    
    norm_hyp = tf.keras.layers.Normalization(mean=hyp_norm[1], variance=hyp_norm[0]**2, axis=-1)(hyp_input)
    
    h = norm_hyp
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation)(h)
        
    outputs = tf.keras.layers.Dense(1, activation=tf.math.softplus, name="acc_dense_out", 
                                    bias_initializer=tf.keras.initializers.Constant(-3.5))(h)
    return tf.keras.Model(inputs=hyp_input, outputs=outputs, name="AcceptanceNet")
