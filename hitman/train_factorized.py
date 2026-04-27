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
            vertex = data['vertex']

            injected_yields = data['scintPhotons'] # TODO: Include cherPhotons in future work once they are properly scaled with respect to light yield and detector sensitivity.
            total_hits = np.sum(shape_targets, axis=1)
            rate_targets = np.stack([total_hits, injected_yields], axis=1)
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
    
    acc_features = np.zeros((len(charge_hyp), 8), dtype=np.float32)
    scat = charge_hyp[:, 0]
    abs_len = charge_hyp[:, 1]
    
    L_fiber_init = 1000.0
    inv_L_eff = (1.0 / (abs_len + 1e-12)) + (1.0 / L_fiber_init)
    L_eff = 1.0 / inv_L_eff
    
    L_D = np.sqrt((L_eff * scat) / 3.0)
    
    Sigma_scat = 1.0 / (scat + 1e-12)
    omega = Sigma_scat / (Sigma_scat + inv_L_eff)
    
    acc_features[:, 0] = np.log(scat + 1e-12)
    acc_features[:, 1] = np.log(abs_len + 1e-12)
    acc_features[:, 2] = np.log(L_eff + 1e-12)
    acc_features[:, 3] = np.log(L_D + 1e-12)
    acc_features[:, 4] = omega
    
    # Apply Z-folding (Symmetry 5) and Transverse Reflection (Symmetries 1-4)
    z_raw = vertex[:, 2]
    condition = z_raw < 0
    
    x_folded = np.where(condition, vertex[:, 1], vertex[:, 0])
    y_folded = np.where(condition, vertex[:, 0], vertex[:, 1])
    z_folded = np.abs(z_raw)
    
    acc_features[:, 5] = np.abs(x_folded) # |X_folded|
    acc_features[:, 6] = np.abs(y_folded) # |Y_folded|
    acc_features[:, 7] = z_folded # |Z|
    
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
            if logs is None:
                return
            val_metric = logs.get(f'val_{self.metric_name}')
            if val_metric is not None:
                std_err = np.sqrt(2.0 / self.total_dof)
                lower = 1.0 - 1.96 * std_err
                upper = 1.0 + 1.96 * std_err
                print(f"\n---> {self.metric_name} Validation: {val_metric:.4f} (95% CI: [{lower:.3f}, {upper:.3f}])")

    class WarmUpCallback(tf.keras.callbacks.Callback):
        def __init__(self, initial_lr, warmup_epochs, start_factor=0.1):
            super(WarmUpCallback, self).__init__()
            self.initial_lr = initial_lr
            self.warmup_epochs = warmup_epochs
            self.start_factor = start_factor
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                progress = epoch / (self.warmup_epochs - 1) if self.warmup_epochs > 1 else 1.0
                new_lr = self.initial_lr * (self.start_factor + (1.0 - self.start_factor) * progress)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'\nWarmup - setting learning rate to {new_lr:.6f}.')

    # ==========================================
    # Train ShapeNet
    # ==========================================
    if not args.acc_only:
        print("\n----- Training ShapeNet -----")
        train_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, vertex, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
        val_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, vertex, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
        
        def multinomial_crossentropy(y_true, y_pred):
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.nn.log_softmax(y_pred), axis=-1))

        def pearson_chi2(y_true, y_pred):
            K_obs = tf.reduce_sum(y_true, axis=-1, keepdims=True)
            p_i = tf.nn.softmax(y_pred, axis=-1)
            mu_i = K_obs * p_i
            chi2 = tf.reduce_sum(tf.math.squared_difference(y_true, mu_i) / (mu_i + 1e-12), axis=-1)
            dof = tf.cast(tf.shape(pmt_positions)[0] - 1, tf.float32)
            return tf.reduce_mean(chi2 / dof)

        with strategy.scope():
            shape_net = get_shape_net(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, acc_hyp_norm=acc_hyp_norm, obs_norm=obs_norm, use_d2h=not args.no_d2h, use_vertex=True)
            optimizer_s = tf.keras.optimizers.Adam(args.lr)
            shape_net.compile(loss=multinomial_crossentropy, optimizer=optimizer_s, metrics=[pearson_chi2])

        callbacks_s = [
            WarmUpCallback(args.lr, warmup_epochs=3),
            DynamicBoundsCallback('pearson_chi2', val_events * (len(pmt_positions) - 1.0)),
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
        
        # Save normalization constants
        np.save(os.path.join(args.output_network, 'hyp_norm.npy'), hyp_norm)
        np.save(os.path.join(args.output_network, 'obs_norm.npy'), obs_norm)
        
        tf.keras.models.save_model(shape_net, os.path.join(args.output_network, 'ShapeNet'), save_format='tf')
    
    # ==========================================
    # Train AcceptanceNet
    # ==========================================
    print("\n----- Training AcceptanceNet -----")
    train_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, vertex, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
    val_gen_acc = get_acceptance_dataset(rate_targets, charge_hyp, vertex, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
    
    def effective_poisson_nll(y_true, y_pred):
        y_true_fp64 = tf.cast(y_true, tf.float64)
        y_pred_fp64 = tf.cast(y_pred, tf.float64)
        
        K_sim = y_true_fp64[:, 0:1]
        eta_sim = y_true_fp64[:, 1:2]
        z_eps = y_pred_fp64
        
        loss = eta_sim * tf.exp(z_eps) - K_sim * z_eps
        return tf.reduce_mean(loss)

    def binomial_deviance(y_true, y_pred):
        y_true_fp64 = tf.cast(y_true, tf.float64)
        y_pred_fp64 = tf.cast(y_pred, tf.float64)
        
        K_sim = y_true_fp64[:, 0:1]
        eta_sim = y_true_fp64[:, 1:2]
        
        z_eps = tf.clip_by_value(y_pred_fp64, -1e6, -1e-6)
        
        term1 = tf.math.xlogy(K_sim, K_sim / eta_sim) - K_sim * z_eps
        
        rem_obs = eta_sim - K_sim
        term2 = tf.math.xlogy(rem_obs, rem_obs / eta_sim) - rem_obs * tf.math.log1p(-tf.exp(z_eps))
        
        D_bin = 2.0 * (term1 + term2)
        return tf.reduce_mean(D_bin)

    with strategy.scope():
        acc_net = get_acceptance_net(layers=args.layers, nodes=args.nodes, hyp_norm=acc_hyp_norm, obs_norm=None)
        optimizer_a = tf.keras.optimizers.Adam(args.lr)
        acc_net.compile(loss=effective_poisson_nll, optimizer=optimizer_a, metrics=[binomial_deviance], jit_compile=True)
        
    callbacks_a = [
        WarmUpCallback(args.lr, warmup_epochs=3),
        DynamicBoundsCallback('binomial_deviance', val_events * 1.0),
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
    
    from hitman.neural_nets.factorized_chargenet import get_wrapped_acceptance_net
    wrapped_acc_net = get_wrapped_acceptance_net(acc_net, use_logsigmoid=use_logsigmoid)
    tf.keras.models.save_model(wrapped_acc_net, os.path.join(args.output_network, 'AcceptanceNet'), save_format='tf')

    # Automatically export weights for pure JAX inference engines
    from hitman.tools.jax_exporter import export_to_jax
    export_to_jax(args.output_network, use_logsigmoid=use_logsigmoid)

if __name__ == '__main__':
    main()
t_to_jax(args.output_network)

if __name__ == '__main__':
    main()
 engines
    from hitman.tools.jax_exporter import export_to_jax
    export_to_jax(args.output_network)

if __name__ == '__main__':
    main()
