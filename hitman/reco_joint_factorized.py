import tensorflow as tf
import numpy as np

@tf.function
def newton_raphson_epsilon(S, Lambda_sig, b, charges, max_iter=20, tol=1e-5):
    N_hyp = tf.shape(S)[0]
    b_broadcast = tf.reshape(b, (1, 1, -1))
    sum_Lambda_sig = tf.reduce_sum(Lambda_sig, axis=1, keepdims=True)
    charges_broadcast = tf.expand_dims(charges, axis=0)

    # FIX: Smart Algebraic Seed
    total_charges = tf.reduce_sum(charges)
    total_b = tf.reduce_sum(b) * tf.cast(tf.shape(charges)[0], tf.float32)
    eps_guess = tf.maximum((total_charges - total_b) / (sum_Lambda_sig + 1e-12), 1e-4)
    eps = eps_guess # Automatically broadcasts to (N_hyp, 1)

    for i in tf.range(max_iter):
        eps_broadcast = tf.expand_dims(eps, axis=-1)
        lam = eps_broadcast * S + b_broadcast
        term1 = tf.reduce_sum(charges_broadcast * S / lam, axis=[1, 2])
        dL_deps = -sum_Lambda_sig + tf.reshape(term1, (N_hyp, 1))
        d2L_deps2 = -tf.reduce_sum(charges_broadcast * (S ** 2) / (lam ** 2), axis=[1, 2])
        step = dL_deps / (tf.reshape(d2L_deps2, (N_hyp, 1)) - 1e-12)
        eps = tf.clip_by_value(eps - step, 1e-5, 10.0)
        if tf.reduce_max(tf.abs(step)) < tol: break
    return eps

@tf.function
def tfLLH_joint_factorized(charges, pmt_positions, hyp_batch, yields, shape_net, acc_net, b_val=0.1, fixed_eps=None):
    N_hyp = tf.shape(hyp_batch)[0]
    N_events = tf.shape(charges)[0]
    N_sensors = tf.shape(charges)[1]
    
    pmt_tiled = tf.tile(tf.expand_dims(pmt_positions, 0), [N_hyp, 1, 1])
    
    # Evaluate ONCE per material hypothesis
    logits_fp64 = shape_net([hyp_batch, pmt_tiled]) # Shape: (N_hyp, N_sensors)
    f_nn = tf.cast(tf.nn.softmax(logits_fp64, axis=1), tf.float32)
    
    mu_nn = acc_net([hyp_batch])             # Shape: (N_hyp, 1)
    
    # mu_nn shape: (N_hyp, 1) -> expand to (N_hyp, 1)
    # yields shape: (N_events,) -> expand to (1, N_events)
    # Lambda_sig broadcasts to (N_hyp, N_events)
    Lambda_sig = mu_nn * tf.expand_dims(yields, axis=0) 

    # Lambda_sig shape: (N_hyp, N_events) -> expand to (N_hyp, N_events, 1)
    # f_nn shape: (N_hyp, N_sensors) -> expand to (N_hyp, 1, N_sensors)
    # S broadcasts automatically to (N_hyp, N_events, N_sensors)
    S = tf.expand_dims(Lambda_sig, axis=-1) * tf.expand_dims(f_nn, axis=1)    
    b = tf.ones((N_sensors,), dtype=tf.float32) * b_val
    
    if fixed_eps is None:
        eps = newton_raphson_epsilon(S, Lambda_sig, b, charges)
    else:
        eps = tf.ones((N_hyp, 1), dtype=tf.float32) * tf.cast(fixed_eps, tf.float32)
        
    lam = tf.expand_dims(eps, axis=-1) * S + tf.reshape(b, (1, 1, N_sensors))
    
    # Standard Poisson LLH: k*log(lam) - lam - log(k!)
    # Using EXACT factorial: ln(k!) = ln(Gamma(k+1))
    charges_broadcast = tf.expand_dims(charges, axis=0)
    llh_matrix = charges_broadcast * tf.math.log(lam) - lam

    exact_log_factorial = tf.math.lgamma(charges_broadcast + 1.0)

    # Only apply where charges > 0
    mask = tf.cast(charges_broadcast > 0, tf.float32)
    llh_matrix = llh_matrix - (exact_log_factorial * mask)

    return -tf.reduce_sum(llh_matrix, axis=[1, 2])
