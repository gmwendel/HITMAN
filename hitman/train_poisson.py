#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import os
from hitman.tools.ratextract_poisson import PoissonDataExtractor
from hitman.tools.poisson_datagenerator import PoissonDataGenerator
from hitman.neural_nets.poisson_chargenet import get_poisson_chargenet, poisson_nll_loss
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input locations', nargs='+', required=True)
    parser.add_argument('-o', '--output_network', help='Output location for network', required=True)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--nodes', default=256, type=int)
    parser.add_argument('--batch_power', default=15, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    return parser.parse_args()

def main():
    args = get_args()
    
    import glob
    expanded_files = []
    for f in args.input_files:
        expanded_files.extend(glob.glob(f))
        
    Data = PoissonDataExtractor(expanded_files)
    charges, charge_hyp, pmt_positions = Data.get_poisson_train_data()
    print("Data Loaded. Events:", len(charges), "Sensors:", len(pmt_positions))
    
    hyp_norm = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    obs_norm = np.stack([np.std(pmt_positions, axis=0), np.mean(pmt_positions, axis=0)])
    
    hyp_norm[0][hyp_norm[0] == 0] = 1.0
    obs_norm[0][obs_norm[0] == 0] = 1.0
    
    splits = max(1, int(len(charges) / 10))
    train_charges, val_charges = charges[:-splits], charges[-splits:]
    train_hyp, val_hyp = charge_hyp[:-splits], charge_hyp[-splits:]
    
    train_gen = PoissonDataGenerator(train_charges, train_hyp, pmt_positions, batch_size=2**args.batch_power)
    val_gen = PoissonDataGenerator(val_charges, val_hyp, pmt_positions, batch_size=2**args.batch_power)
    
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        model = get_poisson_chargenet(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        optimizer = tf.keras.optimizers.Adam(args.lr)
        model.compile(loss=poisson_nll_loss, optimizer=optimizer)
        
    os.makedirs(args.output_network, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
    
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=args.epochs, 
        callbacks=callbacks, 
        verbose=2, 
        workers=16, 
        use_multiprocessing=True,
        max_queue_size=512
    )
    
    tf.keras.models.save_model(model, args.output_network + '/poisson_chargenet', save_format='tf')

if __name__ == '__main__':
    main()