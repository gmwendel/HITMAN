#!/usr/bin/env python3
import os

# Mandate Full FP32 Precision for Physics Rigor
# MUST be set before importing tensorflow
os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '0'

import argparse
import numpy as np
import tensorflow as tf

# Explicitly disable TF32 via API as a double-safeguard
tf.config.experimental.enable_tensor_float_32_execution(False)

import glob
from hitman.tools.ratextract_factorized import FactorizedDataExtractor
from hitman.tools.factorized_datagenerator import get_shape_dataset, get_acceptance_dataset
from hitman.neural_nets.factorized_chargenet import get_shape_net, get_acceptance_net
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input locations', nargs='+', required=True)
    parser.add_argument('-o', '--output_network', help='Output location for network', required=True)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--pretrained_network', required=True)
    parser.add_argument('--chi2_cut', default=5255.0, type=float)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--nodes', default=128, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--no_d2h', action='store_true', help='Disable D2h Symmetrization')
    parser.add_argument('--acc_only', action='store_true', help='Only train AcceptanceNet')
    parser.add_argument('--shape_only', action='store_true', help='Only train ShapeNet')
    parser.add_argument('--shape_layers', type=int, default=None)
    parser.add_argument('--shape_nodes', type=int, default=None)
    parser.add_argument('--acc_layers', type=int, default=None)
    parser.add_argument('--acc_nodes', type=int, default=None)
    parser.add_argument('--subset_fraction', type=float, default=1.0, help='Fraction of dataset to use for training (0.0 to 1.0)')
    parser.add_argument('--fiducial_cut', type=float, default=75.0, help='Maximum absolute X/Y/Z boundary for training events')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Defaults for split architecture
    s_layers = args.shape_layers if args.shape_layers is not None else args.layers
    s_nodes = args.shape_nodes if args.shape_nodes is not None else args.nodes
    a_layers = args.acc_layers if args.acc_layers is not None else args.layers
    a_nodes = args.acc_nodes if args.acc_nodes is not None else args.nodes

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== HITMAN Factorized Training: {timestamp} ===")
    print(f"ShapeNet Arch:      L{s_layers} N{s_nodes}")
    print(f"AcceptanceNet Arch: L{a_layers} N{a_nodes}")
    print(f"Subset:             {args.subset_fraction*100:.1f}%")
    print(f"Precision:          Full FP32 (TF32 Disabled)")
    
    cache_path = args.input_files[0] if args.input_files[0].endswith(".npz") else "data/high_yield_spatial_raw_cache.npz"
    if os.path.exists(cache_path) and cache_path.endswith(".npz"):
        print(f"Loading data instantly from raw cache: {cache_path}")
        with np.load(cache_path) as data:
            shape_targets = data['charges'] # Raw, un-smoothed integer hit counts
            charge_hyp = data['charge_hyp']
            pmt_positions = data['pmt_positions']
            if 'pmt_dirs' in data:
                pmt_dirs = data['pmt_dirs']
            else:
                pmt_dirs = np.zeros_like(pmt_positions)
                d_idx = np.argmax(np.abs(pmt_positions), axis=-1)
                pmt_dirs[np.arange(len(pmt_positions)), d_idx] = np.sign(pmt_positions[np.arange(len(pmt_positions)), d_idx])
            energy = data['energy']
            scint_edep = data['scintEdep'] if 'scintEdep' in data else energy
            vertex = data['vertex']

            injected_yields = data['scintPhotons'] # TODO: Include cherPhotons in future work once they are properly scaled with respect to light yield and detector sensitivity.
            total_hits = np.sum(shape_targets, axis=1)
            rate_targets = np.stack([total_hits, injected_yields], axis=1)
            
            print(f"Loaded {len(shape_targets)} total events from cache. Applying cuts...")
            valid_energy = np.abs(scint_edep - 0.3736) < 0.05
            valid_vtx = np.all(np.abs(vertex) <= args.fiducial_cut, axis=-1)
            
            max_hits = np.max(shape_targets, axis=1)
            hit_frac = max_hits / (total_hits + 1e-12)
            valid_topo = (total_hits > 0) & (hit_frac <= 0.50)
            
            valid_mask = valid_energy & valid_vtx & valid_topo
            
            print(f"Loading pretrained ShapeNet from {args.pretrained_network} to calculate Chi2...")
            pretrained_shape_model = tf.keras.models.load_model(os.path.join(args.pretrained_network, 'ShapeNet'), compile=False)
            
            chunk_size = 5000
            chi2_mask = np.ones(len(shape_targets), dtype=bool)
            
            u_frames = None
            if len(pretrained_shape_model.inputs) == 6:
                with np.load('data/covariant_frames.npz') as f:
                    u_frames = f['u_frames']
                    v_frames = f['v_frames']
                    w_frames = f['w_frames']
            
            pmt_pos_tf = tf.constant(pmt_positions, dtype=tf.float32)
            
            import sys
            for i in range(0, len(shape_targets), chunk_size):
                if i % 100000 == 0:
                    sys.stdout.write(f"\rEvaluating Chi2: {i}/{len(shape_targets)}")
                    sys.stdout.flush()
                    
                end = min(i + chunk_size, len(shape_targets))
                if not valid_mask[i:end].any():
                    chi2_mask[i:end] = False
                    continue
                    
                h_c = tf.constant(charge_hyp[i:end], dtype=tf.float32)
                v_c = tf.constant(vertex[i:end], dtype=tf.float32)
                pmt_c = tf.tile(tf.expand_dims(pmt_pos_tf, 0), [end-i, 1, 1])
                
                if u_frames is not None:
                    u_c = tf.tile(tf.expand_dims(tf.constant(u_frames, dtype=tf.float32), 0), [end-i, 1, 1])
                    v_c_f = tf.tile(tf.expand_dims(tf.constant(v_frames, dtype=tf.float32), 0), [end-i, 1, 1])
                    w_c = tf.tile(tf.expand_dims(tf.constant(w_frames, dtype=tf.float32), 0), [end-i, 1, 1])
                    inputs = [h_c, pmt_c, v_c, u_c, v_c_f, w_c]
                else:
                    inputs = [h_c, pmt_c, v_c]
                    
                logits = pretrained_shape_model(inputs, training=False)
                logits_f64 = tf.cast(logits, tf.float64)
                p_i = tf.nn.softmax(logits_f64, axis=-1)
                
                y_true = tf.cast(shape_targets[i:end], tf.float64)
                K_obs = tf.reduce_sum(y_true, axis=-1, keepdims=True)
                mu_i = K_obs * p_i
                
                chi2 = tf.reduce_sum(tf.square(y_true - mu_i) / (mu_i + 1e-12), axis=-1).numpy()
                chi2_mask[i:end] = chi2 <= args.chi2_cut

            print("")
            final_mask = valid_mask & chi2_mask
            
            del pretrained_shape_model
            tf.keras.backend.clear_session()
            
            shape_targets = shape_targets[final_mask]
            charge_hyp = charge_hyp[final_mask]
            vertex = vertex[final_mask]
            rate_targets = rate_targets[final_mask]
            
            # Apply subset fraction for fast validation
            if args.subset_fraction < 1.0:
                N_total = len(shape_targets)
                N_subset = int(N_total * args.subset_fraction)
                print(f"Reducing dataset to {N_subset} / {N_total} events ({args.subset_fraction*100:.1f}%)")
                subset_idx = np.random.choice(N_total, N_subset, replace=False)
                shape_targets = shape_targets[subset_idx]
                charge_hyp = charge_hyp[subset_idx]
                vertex = vertex[subset_idx]
                rate_targets = rate_targets[subset_idx]

            print(f"Final Data Count: {len(shape_targets)} high-quality training events.")
    else:
        print(f"Cache not found at {cache_path}. Falling back to slow ROOT extraction...")
        expanded_files = []
        for f in args.input_files:
            expanded_files.extend(glob.glob(f))
        expanded_files = sorted(expanded_files)

        Data = FactorizedDataExtractor(expanded_files)
        shape_targets, rate_targets, charge_hyp, pmt_positions, pmt_dirs, vertex = Data.get_factorized_only_train_data()        
    print(f"Data Loaded. Events: {len(charge_hyp)}, Sensors: {len(pmt_positions)}")
    
    # Global shuffle to prevent validation set from being biased to a specific parameter region
    print("Globally shuffling dataset...")
    np.random.seed(42) # For reproducible train/val splits
    shuffle_idx = np.random.permutation(len(charge_hyp))
    
    shape_targets = shape_targets[shuffle_idx]
    rate_targets = rate_targets[shuffle_idx]
    charge_hyp = charge_hyp[shuffle_idx]
    vertex = vertex[shuffle_idx]
    
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
    
    acc_features = np.zeros((len(charge_hyp), 10))

    # Calculate Fiber Physics analytically
    scat = charge_hyp[:, 0]
    abs_len = charge_hyp[:, 1]
    
    # Physics Priors bypassed. Use raw hypothesis parameters.
    acc_features = np.zeros((len(charge_hyp), 7), dtype=np.float32)
    
    acc_features[:, 0] = np.log(scat + 1e-12)
    acc_features[:, 1] = np.log(abs_len + 1e-12)
    
    # Apply D2d Harmonic Invariants
    detector_scale = 150.0
    x_n = vertex[:, 0] / detector_scale
    y_n = vertex[:, 1] / detector_scale
    z_n = vertex[:, 2] / detector_scale
    
    acc_features[:, 2] = x_n**2 + y_n**2
    acc_features[:, 3] = x_n**4 + y_n**4
    acc_features[:, 4] = z_n**2
    acc_features[:, 5] = (x_n**2 - y_n**2) * z_n
    acc_features[:, 6] = (x_n * y_n * z_n)**2
    
    acc_hyp_norm = np.stack([np.std(acc_features, axis=0), np.mean(acc_features, axis=0)])
    acc_hyp_norm[0][acc_hyp_norm[0] == 0] = 1.0
    print(f"Acc Hypothesis Norm - Mean: {acc_hyp_norm[1]}, Std: {acc_hyp_norm[0]}")
    
    N_events = len(charge_hyp)
    val_events = max(1, int(N_events * 0.1))
    train_events = N_events - val_events
    
    steps_train = max(1, int(train_events / args.batch_size))
    steps_val = max(1, int(val_events / args.batch_size))
    
    class DynamicBoundsCallback(tf.keras.callbacks.Callback):
        def __init__(self, metric_name, total_dof):
            super(DynamicBoundsCallback, self).__init__()
            self.metric_name = metric_name
            self.total_dof = total_dof

        def on_epoch_end(self, epoch, logs=None):
            val_metric = logs.get(f'val_{self.metric_name}')
            if val_metric is not None:
                # Calculate expected Chi2 range for 95% confidence
                # For high DOF, Chi2 ~ N(dof, sqrt(2*dof))
                mean = 1.0
                std = np.sqrt(2.0 / self.total_dof)
                lower = mean - 2*std
                upper = mean + 2*std
                print(f"\n---> {self.metric_name} Validation: {val_metric:.4f} (95% CI: [{lower:.3f}, {upper:.3f}])")

    class WarmUpCallback(tf.keras.callbacks.Callback):
        def __init__(self, initial_lr, warmup_epochs=5):
            super(WarmUpCallback, self).__init__()
            self.initial_lr = initial_lr
            self.warmup_epochs = warmup_epochs

        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                print(f"\nWarmup - setting learning rate to {lr:f}.")

    class ShapeDiagnosticsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_dataset, steps):
            super(ShapeDiagnosticsCallback, self).__init__()
            self.val_dataset = val_dataset
            self.steps = steps

        def on_epoch_end(self, epoch, logs=None):
            total_D = 0.0
            pull_sum = 0.0
            pull_sq_sum = 0.0
            pull_count = 0.0
            
            high_yield_chi2 = 0.0
            high_yield_count = 0.0
            
            dof = None
            events_processed = 0
            
            for i, (x_batch, y_true_batch) in enumerate(self.val_dataset.take(self.steps)):
                y_pred_batch = self.model.predict(x_batch, verbose=0)
                
                y_true = tf.cast(y_true_batch, tf.float64)
                y_pred = tf.cast(y_pred_batch, tf.float64)
                
                K_obs = tf.reduce_sum(y_true, axis=-1, keepdims=True)
                p_i = tf.nn.softmax(y_pred, axis=-1)
                mu_i = K_obs * p_i
                
                if dof is None:
                    dof = tf.cast(tf.shape(y_true)[1] - 1, tf.float64)
                
                # Poisson Deviance: 2 * k_i * ln(k_i / mu_i)
                term1 = tf.math.xlogy(y_true, y_true / (mu_i + 1e-12))
                D_batch = 2.0 * tf.reduce_sum(term1, axis=-1)
                total_D += tf.reduce_sum(D_batch).numpy()
                
                # Pull Tensor Distributions
                Z_i = (y_true - mu_i) / tf.sqrt(mu_i + 1e-12)
                pull_sum += tf.reduce_sum(Z_i).numpy()
                pull_sq_sum += tf.reduce_sum(tf.square(Z_i)).numpy()
                pull_count += tf.size(Z_i).numpy()
                
                # High-Yield Stratification
                high_yield_mask = mu_i > 5.0

                if tf.reduce_any(high_yield_mask):
                    y_true_hy = tf.boolean_mask(y_true, high_yield_mask)
                    mu_i_hy = tf.boolean_mask(mu_i, high_yield_mask)

                    chi2_hy = tf.reduce_sum(tf.square(y_true_hy - mu_i_hy) / (mu_i_hy + 1e-12))
                    high_yield_chi2 += chi2_hy.numpy()
                    high_yield_count += tf.cast(tf.size(y_true_hy), tf.float64).numpy()                    
                events_processed += y_true.shape[0]

            mean_D_per_dof = (total_D / events_processed) / dof.numpy()
            
            pull_mean = pull_sum / pull_count
            pull_std = np.sqrt((pull_sq_sum / pull_count) - pull_mean**2)
            
            print(f"\n--- ShapeNet Diagnostics ---")
            print(f"Poisson Deviance / dof : {mean_D_per_dof:.4f}")
            print(f"Pull Tensor            : Mean = {pull_mean:.4f}, StdDev = {pull_std:.4f}")
            
            if high_yield_count > 0:
                mean_hy_chi2_per_dof = high_yield_chi2 / high_yield_count
                print(f"High-Yield Pearson x2/dof : {mean_hy_chi2_per_dof:.4f} (across {int(high_yield_count)} bright bins)")
            print(f"----------------------------")

    # ==========================================
    # Train ShapeNet
    # ==========================================
    if not args.acc_only:
        print("\n----- Training ShapeNet -----")
        u_frames = None # Optional: Set to data['pmt_dirs'] if using dirNet
        train_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, vertex, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1, u_frames=u_frames)
        val_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, vertex, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1, u_frames=u_frames)

        def multinomial_crossentropy(y_true, y_pred):
            y_true_fp64 = tf.cast(y_true, tf.float64)
            y_pred_fp64 = tf.cast(y_pred, tf.float64)
            logits = y_pred_fp64 - tf.reduce_logsumexp(y_pred_fp64, axis=-1, keepdims=True)
            loss = -tf.reduce_sum(y_true_fp64 * logits, axis=-1)
            return tf.reduce_mean(loss)

        def poisson_deviance(y_true, y_pred):
            y_true_fp64 = tf.cast(y_true, tf.float64)
            y_pred_fp64 = tf.cast(y_pred, tf.float64)
            K_obs = tf.reduce_sum(y_true_fp64, axis=-1, keepdims=True)
            p_i = tf.nn.softmax(y_pred_fp64, axis=-1)
            mu_i = K_obs * p_i
            term1 = tf.math.xlogy(y_true_fp64, y_true_fp64 / (mu_i + 1e-12))
            D = 2.0 * tf.reduce_sum(term1, axis=-1)
            dof = tf.cast(tf.shape(pmt_positions)[0] - 1, tf.float64)
            return tf.reduce_mean(D / dof)

        with strategy.scope():
            shape_net = tf.keras.models.load_model(os.path.join(args.pretrained_network, 'ShapeNet'), compile=False)
            optimizer_s = tf.keras.optimizers.Adam(args.lr)
            shape_net.compile(loss=multinomial_crossentropy, optimizer=optimizer_s, metrics=[poisson_deviance])

        callbacks_s = [
            ShapeDiagnosticsCallback(val_gen_shape, steps_val),
            WarmUpCallback(args.lr, warmup_epochs=3),
            DynamicBoundsCallback('poisson_deviance', val_events * (len(pmt_positions) - 1.0)),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
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
        np.save(os.path.join(args.output_network, 'hyp_norm.npy'), hyp_norm)
        np.save(os.path.join(args.output_network, 'obs_norm.npy'), obs_norm)
        tf.keras.models.save_model(shape_net, os.path.join(args.output_network, 'ShapeNet'), save_format='tf')
    
    # ==========================================
    # Train AcceptanceNet
    # ==========================================
    if not args.shape_only:
        print("\n----- Training AcceptanceNet -----")
        train_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, vertex, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
        val_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, vertex, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
        
        def effective_poisson_nll(y_true, y_pred):
            y_true_fp64 = tf.cast(y_true, tf.float64)
            y_pred_fp64 = tf.cast(y_pred, tf.float64)
            K_sim = y_true_fp64[:, 0:1]
            eta_sim = y_true_fp64[:, 1:2]
            z_eps = tf.math.log_sigmoid(y_pred_fp64)
            loss = eta_sim * tf.exp(z_eps) - K_sim * z_eps
            return tf.reduce_mean(loss)

        def binomial_deviance(y_true, y_pred):
            y_true_fp64 = tf.cast(y_true, tf.float64)
            y_pred_fp64 = tf.cast(y_pred, tf.float64)
            K_sim = y_true_fp64[:, 0:1]
            eta_sim = y_true_fp64[:, 1:2]
            log_p = tf.math.log_sigmoid(y_pred_fp64)
            log_1_minus_p = tf.math.log_sigmoid(-y_pred_fp64)
            term1 = tf.math.xlogy(K_sim, K_sim / eta_sim) - K_sim * log_p
            rem_obs = eta_sim - K_sim
            term2 = tf.math.xlogy(rem_obs, rem_obs / eta_sim) - rem_obs * log_1_minus_p
            return tf.reduce_mean(2.0 * (term1 + term2))

        with strategy.scope():
            acc_net = tf.keras.models.load_model(os.path.join(args.pretrained_network, 'AcceptanceNet'), compile=False)
            optimizer_a = tf.keras.optimizers.Adam(args.lr)
            acc_net.compile(loss=effective_poisson_nll, optimizer=optimizer_a, metrics=[binomial_deviance])

        callbacks_a = [
            WarmUpCallback(args.lr, warmup_epochs=3),
            DynamicBoundsCallback('binomial_deviance', val_events),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
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
        tf.keras.models.save_model(acc_net, os.path.join(args.output_network, 'AcceptanceNet'), save_format='tf')

    # Export to JAX
    print("\n--- Exporting TF models to JAX-compatible format ---")
    weights = {}
    if not args.acc_only:
        print("Exporting ShapeNet weights...")
        s_model = tf.keras.models.load_model(os.path.join(args.output_network, 'ShapeNet'), compile=False)
        for layer in s_model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[f"shape_{layer.name}_0"] = w[0]
                if len(w) > 1:
                    weights[f"shape_{layer.name}_1"] = w[1]
    
    if not args.shape_only:
        print("Exporting AcceptanceNet weights...")
        a_model = tf.keras.models.load_model(os.path.join(args.output_network, 'AcceptanceNet'), compile=False)
        for layer in a_model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[f"acc_{layer.name}_0"] = w[0]
                if len(w) > 1:
                    weights[f"acc_{layer.name}_1"] = w[1]

    # Explicitly map the normalization layers which don't export cleanly via get_weights
    acc_norm = np.load(os.path.join(args.output_network, 'acc_hyp_norm.npy'))
    weights['acc_normalization_mean'] = acc_norm[1]
    weights['acc_normalization_var'] = acc_norm[0]**2
    weights['shape_normalization_mean'] = acc_norm[1][:2]
    weights['shape_normalization_var'] = acc_norm[0][:2]**2

    np.savez_compressed(os.path.join(args.output_network, 'jax_weights.npz'), **weights)
    import json
    with open(os.path.join(args.output_network, 'config.json'), 'w') as f:
        json.dump({"use_logsigmoid": True}, f)
    print(f"Saved JAX weights and config to {args.output_network}")

if __name__ == "__main__":
    main()
