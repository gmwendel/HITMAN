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
    return parser.parse_args()

def main():
    args = get_args()
    
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
    
    with strategy.scope():
        shape_net = get_shape_net(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        optimizer_s = tf.keras.optimizers.Adam(args.lr)
        # Using KL Divergence for PMF prediction
        shape_net.compile(loss=tf.keras.losses.KLDivergence(), optimizer=optimizer_s)
        
    callbacks_s = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    shape_net.fit(
        train_gen_shape, 
        validation_data=val_gen_shape, 
        epochs=args.epochs, 
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        callbacks=callbacks_s, 
        verbose=2
    )
    
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
        # Mean Absolute Percentage Error ensures precise relative accuracy on tiny (0.01-0.05) acceptance fractions
        acc_net.compile(loss='mape', optimizer=optimizer_a)
        
    callbacks_a = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    acc_net.fit(
        train_gen_acc, 
        validation_data=val_gen_acc, 
        epochs=args.epochs, 
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        callbacks=callbacks_a, 
        verbose=2
    )
    
    tf.keras.models.save_model(acc_net, os.path.join(args.output_network, 'AcceptanceNet'), save_format='tf')

if __name__ == '__main__':
    main()
