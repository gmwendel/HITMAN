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
            rate_targets = total_hits / injected_yields
    else:
        print(f"Cache not found at {cache_path}. Falling back to slow ROOT extraction...")
        expanded_files = []
        for f in args.input_files:
            expanded_files.extend(glob.glob(f))
        expanded_files = sorted(expanded_files)
            
        Data = FactorizedDataExtractor(expanded_files)
        shape_targets, rate_targets, charge_hyp, pmt_positions = Data.get_factorized_only_train_data()
        
    print(f"Data Loaded. Events: {len(charge_hyp)}, Sensors: {len(pmt_positions)}")
    
    os.makedirs(args.output_network, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    hyp_norm = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    obs_norm = np.stack([np.std(pmt_positions, axis=0), np.mean(pmt_positions, axis=0)])
    
    hyp_norm[0][hyp_norm[0] == 0] = 1.0
    obs_norm[0][obs_norm[0] == 0] = 1.0

    print(f"Hypothesis Norm - Mean: {hyp_norm[1]}, Std: {hyp_norm[0]}")
    print(f"Observation Norm - Mean: {obs_norm[1]}, Std: {obs_norm[0]}")
    
    N_events = len(charge_hyp)
    val_events = max(1, int(N_events * 0.1))
    train_events = N_events - val_events
    
    steps_train = max(1, int(train_events / args.batch_size))
    steps_val = max(1, int(val_events / args.batch_size))
    
    # ==========================================
    # Train ShapeNet
    # ==========================================
    print("\n----- Training ShapeNet -----")
    train_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, batch_size=args.batch_size, shuffle=True, split='train', val_fraction=0.1)
    val_gen_shape = get_shape_dataset(shape_targets, charge_hyp, pmt_positions, batch_size=args.batch_size, shuffle=False, split='val', val_fraction=0.1)
    
    def multinomial_crossentropy(y_true, y_pred):
        # We manually compute -sum(hits * log_softmax(logits)) to avoid any Keras internal target normalization
        # when hits don't sum to 1. This evaluates the LogSumExp mathematically optimally.
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.nn.log_softmax(y_pred), axis=-1))

    with strategy.scope():
        shape_net = get_shape_net(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm, use_d2h=not args.no_d2h)
        optimizer_s = tf.keras.optimizers.Adam(args.lr)
        shape_net.compile(loss=multinomial_crossentropy, optimizer=optimizer_s)
        
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
    
    with strategy.scope():
        acc_net = get_acceptance_net(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=None)
        optimizer_a = tf.keras.optimizers.Adam(args.lr)
        # Restore Poisson loss as it was likely performing better with the true Poisson nature of photons.
        acc_net.compile(loss=tf.keras.losses.Poisson(), optimizer=optimizer_a)
        
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
    
    tf.keras.models.save_model(acc_net, os.path.join(args.output_network, 'AcceptanceNet'), save_format='tf')

if __name__ == '__main__':
    main()
