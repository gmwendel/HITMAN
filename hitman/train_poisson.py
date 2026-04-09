#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import os
import glob
from hitman.tools.ratextract_poisson import PoissonDataExtractor
from hitman.tools.poisson_datagenerator import PoissonDataGenerator
from hitman.tools.datagenerator import DataGenerator
from hitman.neural_nets.poisson_chargenet import get_poisson_chargenet, poisson_nll_loss
from hitman.neural_nets.hitnet import get_hitnet
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input locations', nargs='+', required=True)
    parser.add_argument('-o', '--output_network', help='Output location for network', required=True)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--nodes', default=128, type=int)
    parser.add_argument('--batch_power', default=15, type=int)
    parser.add_argument('--batch_power_hitnet', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    return parser.parse_args()

def main():
    args = get_args()
    
    expanded_files = []
    for f in args.input_files:
        expanded_files.extend(glob.glob(f))
    expanded_files = sorted(expanded_files)
        
    Data = PoissonDataExtractor(expanded_files)
    charges, charge_hyp, pmt_positions, hit_obs, hit_hyp = Data.get_poisson_train_data()
    print(f"Data Loaded. Events: {len(charges)}, Sensors: {len(pmt_positions)}, Hits: {len(hit_obs)}")
    
    os.makedirs(args.output_network, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    # ==========================================
    # Train Poisson ChargeNet
    # ==========================================
    print("----- Training Poisson ChargeNet -----")
    hyp_norm_charge = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    obs_norm_charge = np.stack([np.std(pmt_positions, axis=0), np.mean(pmt_positions, axis=0)])
    
    hyp_norm_charge[0][hyp_norm_charge[0] == 0] = 1.0
    obs_norm_charge[0][obs_norm_charge[0] == 0] = 1.0
    
    splits_c = max(1, int(len(charges) / 10))
    train_charges, val_charges = charges[:-splits_c], charges[-splits_c:]
    train_charge_hyp, val_charge_hyp = charge_hyp[:-splits_c], charge_hyp[-splits_c:]
    
    train_gen_c = PoissonDataGenerator(train_charges, train_charge_hyp, pmt_positions, batch_size=2**args.batch_power)
    val_gen_c = PoissonDataGenerator(val_charges, val_charge_hyp, pmt_positions, batch_size=2**args.batch_power)
    
    with strategy.scope():
        chargenet = get_poisson_chargenet(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm_charge, obs_norm=obs_norm_charge)
        optimizer_c = tf.keras.optimizers.Adam(args.lr)
        chargenet.compile(loss=poisson_nll_loss, optimizer=optimizer_c)
        
    class StopOnMinLRC(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.lr.numpy()
            if lr <= 0.000126:
                print(f"\\nEpoch {epoch+1}: Learning rate reached minimum threshold ({lr}). Stopping training.")
                self.model.stop_training = True
                
    callbacks_c = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        StopOnMinLRC()
    ]
    
    chargenet.fit(
        train_gen_c, 
        validation_data=val_gen_c, 
        epochs=args.epochs, 
        callbacks=callbacks_c, 
        verbose=2, 
        workers=16, 
        use_multiprocessing=True,
        max_queue_size=512
    )
    tf.keras.models.save_model(chargenet, args.output_network + '/poisson_chargenet', save_format='tf')

    # ==========================================
    # Train HitNet (PerDOM Shuffling)
    # ==========================================
    print("\\n----- Training HitNet (inDOM) -----")
    hyp_norm_hit = np.stack([np.std(hit_hyp, axis=0), np.mean(hit_hyp, axis=0)])
    obs_norm_hit = np.stack([np.std(hit_obs, axis=0), np.mean(hit_obs, axis=0)])
    
    hyp_norm_hit[0][hyp_norm_hit[0] == 0] = 1.0
    obs_norm_hit[0][obs_norm_hit[0] == 0] = 1.0
    
    splits_h = max(1, int(len(hit_obs) / 10))
    train_hits, val_hits = hit_obs[:-splits_h], hit_obs[-splits_h:]
    train_hit_hyp, val_hit_hyp = hit_hyp[:-splits_h], hit_hyp[-splits_h:]
    
    # CRITICAL: shuffle='inDOM'
    train_gen_h = DataGenerator(train_hits, train_hit_hyp, batch_size=2**args.batch_power_hitnet, shuffle='inDOM', time_spread=50)
    val_gen_h = DataGenerator(val_hits, val_hit_hyp, batch_size=2**args.batch_power_hitnet, shuffle='inDOM', time_spread=50)
    
    with strategy.scope():
        hitnet = get_hitnet(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm_hit, obs_norm=obs_norm_hit)
        optimizer_h = tf.keras.optimizers.Adam(args.lr)
        hitnet.compile(loss='binary_crossentropy', optimizer=optimizer_h, metrics=['accuracy'])
        
    class StopOnMinLRH(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.lr.numpy()
            if lr <= 0.000126:
                print(f"\\nEpoch {epoch+1}: Learning rate reached minimum threshold ({lr}). Stopping training.")
                self.model.stop_training = True
                
    callbacks_h = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        StopOnMinLRH()
    ]
    
    hitnet.fit(
        train_gen_h, 
        validation_data=val_gen_h, 
        epochs=args.epochs, 
        callbacks=callbacks_h, 
        verbose=2, 
        workers=16, 
        use_multiprocessing=True,
        max_queue_size=512
    )
    tf.keras.models.save_model(hitnet, args.output_network + '/hitnet', save_format='tf')

if __name__ == '__main__':
    main()