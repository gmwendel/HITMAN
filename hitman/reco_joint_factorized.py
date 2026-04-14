import tensorflow as tf
import numpy as np

@tf.function
def newton_raphson_epsilon(S, Lambda_sig, b, charges, max_iter=20, tol=1e-5):
    """
    1D Newton-Raphson solver for global array efficiency (epsilon).
    
    S: Shape (N_hyp, N_events, N_sensors) - Pure geometric signal tensor (Y * mu * f)
    Lambda_sig: Shape (N_hyp, N_events) - Expected absolute hits (Y * mu)
    b: Shape (N_sensors,) - Independent dark noise per channel
    charges: Shape (N_events, N_sensors) - Observed hits (k)
    """
    # Sum over events:
    # dL/deps = -sum_j Lambda_sig,j + sum_{i,j} (k_{i,j} * S_{i,j} / (eps * S_{i,j} + b_i))
    # d^2L/deps^2 = -sum_{i,j} k_{i,j} * (S_{i,j})^2 / (eps * S_{i,j} + b_i)^2
    
    N_hyp = tf.shape(S)[0]
    eps = tf.ones((N_hyp, 1), dtype=tf.float32) * 0.1 # Seed
    
    # Broadcast b to (1, 1, N_sensors) to match S
    b_broadcast = tf.reshape(b, (1, 1, -1))
    
    # sum_j Lambda_sig_j -> shape (N_hyp, 1)
    sum_Lambda_sig = tf.reduce_sum(Lambda_sig, axis=1, keepdims=True)
    
    # charges: shape (1, N_events, N_sensors)
    charges_broadcast = tf.expand_dims(charges, axis=0)

    for i in tf.range(max_iter):
        eps_broadcast = tf.expand_dims(eps, axis=-1) # (N_hyp, 1, 1)
        
        # lambda = eps * S + b
        lam = eps_broadcast * S + b_broadcast
        
        # first derivative
        # sum_{i,j} (k_{i,j} * S_{i,j} / lam_{i,j})
        term1 = tf.reduce_sum(charges_broadcast * S / lam, axis=[1, 2])
        term1 = tf.reshape(term1, (N_hyp, 1))
        dL_deps = -sum_Lambda_sig + term1
        
        # second derivative
        # -sum_{i,j} k_{i,j} * (S_{i,j})^2 / lam_{i,j}^2
        d2L_deps2 = -tf.reduce_sum(charges_broadcast * (S ** 2) / (lam ** 2), axis=[1, 2])
        d2L_deps2 = tf.reshape(d2L_deps2, (N_hyp, 1))
        
        # Safe update
        step = dL_deps / (d2L_deps2 - 1e-12) # Add small epsilon to prevent div by zero
        eps = eps - step
        eps = tf.clip_by_value(eps, 1e-5, 10.0) # Ensure efficiency stays physically meaningful
        
        # Convergence check
        if tf.reduce_max(tf.abs(step)) < tol:
            break
            
    return eps

@tf.function
def tfLLH_joint_factorized(charges, pmt_positions, hyp_batch, yields, shape_net, acc_net, b_val=0.1, fixed_eps=None):
    """
    Calculates the unbinned extended log-likelihood over a stacked batch of events.
    """
    N_hyp = tf.shape(hyp_batch)[0]
    N_events = tf.shape(charges)[0]
    N_sensors = tf.shape(charges)[1]
    
    # We need to broadcast the hypothesis and PMT positions for the networks
    pmt_broadcast = tf.expand_dims(pmt_positions, axis=0) # (1, N_sensors, 3)
    pmt_tiled = tf.tile(pmt_broadcast, [N_hyp, 1, 1]) # (N_hyp, N_sensors, 3)
    
    # f_NN: (N_hyp, N_sensors)
    f_nn = shape_net([hyp_batch, pmt_tiled])
    
    # mu_NN: (N_hyp, 1)
    mu_nn = acc_net([hyp_batch])
    
    # Expand shapes to accommodate N_events
    f_nn_exp = tf.expand_dims(f_nn, axis=1)
    f_nn_events = tf.tile(f_nn_exp, [1, N_events, 1])
    
    mu_nn_events = tf.tile(mu_nn, [1, N_events])
    
    yields_exp = tf.expand_dims(yields, axis=0)
    
    # Y_j * mu: (N_hyp, N_events)
    Lambda_sig = yields_exp * mu_nn_events
    
    Lambda_sig_exp = tf.expand_dims(Lambda_sig, axis=-1)
    
    # S = Lambda_sig_exp * f_nn_events -> (N_hyp, N_events, N_sensors)
    S = Lambda_sig_exp * f_nn_events
    
    b = tf.ones((N_sensors,), dtype=tf.float32) * b_val
    
    # --- Step A: Profile Epsilon ---
    if fixed_eps is None:
        eps = newton_raphson_epsilon(S, Lambda_sig, b, charges)
    else:
        eps = tf.ones((N_hyp, 1), dtype=tf.float32) * fixed_eps
        
    # Evaluate LLH
    eps_broadcast = tf.expand_dims(eps, axis=-1) # (N_hyp, 1, 1)
    b_broadcast = tf.reshape(b, (1, 1, N_sensors))
    
    lam = eps_broadcast * S + b_broadcast
    
    charges_broadcast = tf.expand_dims(charges, axis=0) # (1, N_events, N_sensors)
    
    llh_matrix = charges_broadcast * tf.math.log(lam) - lam
    
    # Sum over events and sensors to get the total log-likelihood per hypothesis
    llhs = tf.reduce_sum(llh_matrix, axis=[1, 2])
    
    return -llhs
