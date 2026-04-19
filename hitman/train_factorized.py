#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import os
import glob
from hitman.tools.ratextract_factorized import FactorizedDataExtractor
from hitman.tools.factorized_datagenerator import get_shape_dataset, get_acceptance_dataset
from hitman.neural_nets.factorized_chargenet import get_shape_net, get_acceptance_net
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input locations', nargs='+', required=True)
    parser.add_argument('-o', '--output_network', help='Output location for network', required=True)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--nodes', default=128, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--no_d2h', action='store_true', help='Disable D2h Symmetrization')
    parser.add_argument('--acc_only', action='store_true', help='Only train AcceptanceNet')
    return parser.parse_args()

def main():
    args = get_args()
    
    cache_path = "data/high_yield_spatial_raw_cache.npz"
    if os.path.exists(cache_path):
        print(f"Loading data instantly from raw cache: {cache_path}")
        with np.load(cache_path) as data:
            shape_targets = data['charges'] # Raw, un-smoothed integer hit counts
            charge_hyp = data['charge_hyp']
            pmt_positions = data['pmt_positions']
            energy = data['energy']
            
            injected_yields = energy * 100000.0
            total_hits = np.sum(shape_targets, axis=1)
            rate_targets = np.stack([total_hits, injected_yields], axis=1)
    else:
        print(f"Cache not found at {cache_path}. Falling back to slow ROOT extraction...")
        expanded_files = []
        for f in args.input_files:
            expanded_files.extend(glob.glob(f))
        expanded_files = sorted(expanded_files)
            
        Data = FactorizedDataExtractor(expanded_files)
        shape_targets, rate_targets, charge_hyp, pmt_positions = Data.get_factorized_only_train_data()
        
    print(f"Data Loaded. Events: {len(charge_hyp)}, Sensors: {len(pmt_positions)}")
    
    # Global shuffle to prevent validation set from being biased to a specific parameter region
    print("Globally shuffling dataset...")
    np.random.seed(42) # For reproducible train/val splits
    shuffle_idx = np.random.permutation(len(charge_hyp))
    shape_targets = shape_targets[shuffle_idx]
    rate_targets = rate_targets[shuffle_idx]
    charge_hyp = charge_hyp[shuffle_idx]
    
    os.makedirs(args.output_network, exist_ok=True)
    # Save the exact validation indices (first 10% of the shuffled array)
    val_size = max(1, int(len(charge_hyp) * 0.1))
    np.save(os.path.join(args.output_network, 'val_indices.npy'), shuffle_idx[:val_size])
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    hyp_norm = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    obs_norm = np.stack([np.std(pmt_positions, axis=0), np.mean(pmt_positions, axis=0)])
    
    hyp_norm[0][hyp_norm[0] == 0] = 1.0
    obs_norm[0][obs_norm[0] == 0] = 1.0

    print(f"Hypothesis Norm - Mean: {hyp_norm[1]}, Std: {hyp_norm[0]}")
    print(f"Observation Norm - Mean: {obs_norm[1]}, Std: {obs_norm[0]}")
    
    acc_features = np.zeros((len(charge_hyp), 4), dtype=np.float32)
    scat = charge_hyp[:, 0]
    abs_len = charge_hyp[:, 1]
    L_D = np.sqrt((abs_len * scat) / 3.0)
    omega = abs_len / (abs_len + scat + 1e-12)
    
    acc_features[:, 0] = np.log(scat + 1e-12)
    acc_features[:, 1] = np.log(abs_len + 1e-12)
    acc_features[:, 2] = np.log(L_D + 1e-12)
    acc_features[:, 3] = omega
    
    acc_hyp_norm = np.stack([np.std(acc_features, axis=0), np.mean(acc_features, axis=0)])
    acc_hyp_norm[0][acc_hyp_norm[0] == 0] = 1.0
    print(f"Acc Hypothesis Norm - Mean: {acc_hyp_norm[1]}, Std: {acc_hyp_norm[0]}")
    
    N_events = len(charge_hyp)
    val_events = max(1, int(N_events * 0.1))
    train_events = N_events - val_events
    
    steps_train = max(1, int(train_events / args.batch_size))
    steps_val = max(1, int(val_events / args.batch_size))
    
    # ==========================================
    # Train ShapeNet
    # ==========================================
    if not args.acc_only:
        print("\n----- Training ShapeNet -----")
        train_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
        val_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
        
        def multinomial_crossentropy(y_true, y_pred):
            # We manually compute -sum(hits * log_softmax(logits)) to avoid any Keras internal target normalization
            # when hits don't sum to 1. This evaluates the LogSumExp mathematically optimally.
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.nn.log_softmax(y_pred), axis=-1))

        def multinomial_deviance(y_true, y_pred):
            # D_shape = 2 * sum(k_i * ln(k_i / mu_i))
            # mu_i = K_obs * p_i
            K_obs = tf.reduce_sum(y_true, axis=-1, keepdims=True)
            # Prevent division by zero if K_obs=0 (though our dataset filters these)
            K_obs_safe = tf.where(K_obs == 0, tf.ones_like(K_obs), K_obs)
            p_i = tf.nn.softmax(y_pred, axis=-1)
            mu_i = K_obs_safe * p_i
            
            # We must use tf.math.xlogy to safely handle k_i=0
            # tf.math.xlogy(x, y) = x * ln(y), returning 0 if x=0.
            term = tf.math.xlogy(y_true, y_true / (mu_i + 1e-12))
            D_shape = 2.0 * tf.reduce_sum(term, axis=-1)
            return tf.reduce_mean(D_shape)

        with strategy.scope():
            shape_net = get_shape_net(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm, use_d2h=not args.no_d2h)
            optimizer_s = tf.keras.optimizers.Adam(args.lr)
            shape_net.compile(loss=multinomial_crossentropy, optimizer=optimizer_s, metrics=[multinomial_deviance])
            
        callbacks_s = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        
        history_s = shape_net.fit(
            train_gen_shape, 
            validation_data=val_gen_shape, 
            epochs=args.epochs, 
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            callbacks=callbacks_s, 
            verbose=2
        )
        
        np.save(os.path.join(args.output_network, 'shape_training_history.npy'), history_s.history)
        
        # Save normalization constants
        np.save(os.path.join(args.output_network, 'hyp_norm.npy'), hyp_norm)
        np.save(os.path.join(args.output_network, 'obs_norm.npy'), obs_norm)
        
        tf.keras.models.save_model(shape_net, os.path.join(args.output_network, 'ShapeNet'), save_format='tf')
    
    # ==========================================
    # Train AcceptanceNet
    # ==========================================
    print("\n----- Training AcceptanceNet -----")
    train_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
    val_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
    
    def effective_poisson_nll(y_true, y_pred):
        y_true_fp64 = tf.cast(y_true, tf.float64)
        y_pred_fp64 = tf.cast(y_pred, tf.float64)
        
        K_sim = y_true_fp64[:, 0:1]
        eta_sim = y_true_fp64[:, 1:2]
        z_eps = y_pred_fp64
        
        loss = eta_sim * tf.exp(z_eps) - K_sim * z_eps
        return tf.reduce_mean(loss)

    def poisson_deviance(y_true, y_pred):
        # D_coll = 2 * (Lambda_pred - K_obs + K_obs * ln(K_obs / Lambda_pred))
        y_true_fp64 = tf.cast(y_true, tf.float64)
        y_pred_fp64 = tf.cast(y_pred, tf.float64)
        
        K_sim = y_true_fp64[:, 0:1]
        eta_sim = y_true_fp64[:, 1:2]
        z_eps = y_pred_fp64
        
        Lambda_pred = eta_sim * tf.exp(z_eps)
        
        # We must use tf.math.xlogy to safely handle K_sim=0
        # tf.math.xlogy(x, y) = x * ln(y), returning 0 if x=0.
        term = tf.math.xlogy(K_sim, K_sim / (Lambda_pred + 1e-12))
        D_coll = 2.0 * (Lambda_pred - K_sim + term)
        return tf.reduce_mean(D_coll)

    with strategy.scope():
        acc_net = get_acceptance_net(layers=args.layers, nodes=args.nodes, hyp_norm=acc_hyp_norm, obs_norm=None)
        optimizer_a = tf.keras.optimizers.Adam(args.lr)
        acc_net.compile(loss=effective_poisson_nll, optimizer=optimizer_a, metrics=[poisson_deviance])
        
    callbacks_a = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    history_a = acc_net.fit(
        train_gen_acc, 
        validation_data=val_gen_acc, 
        epochs=args.epochs, 
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        callbacks=callbacks_a, 
        verbose=2
    )
    
    np.save(os.path.join(args.output_network, 'acc_training_history.npy'), history_a.history)
    np.save(os.path.join(args.output_network, 'acc_hyp_norm.npy'), acc_hyp_norm)
    
    from hitman.neural_nets.factorized_chargenet import get_wrapped_acceptance_net
    wrapped_acc_net = get_wrapped_acceptance_net(acc_net)
    tf.keras.models.save_model(wrapped_acc_net, os.path.join(args.output_network, 'AcceptanceNet'), save_format='tf')

if __name__ == '__main__':
    main()
