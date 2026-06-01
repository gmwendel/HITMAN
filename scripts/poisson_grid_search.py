#!/usr/bin/env python3
import os
import glob
import itertools
import numpy as np
import tensorflow as tf
import datetime
from hitman.tools.ratextract_poisson import PoissonDataExtractor
from hitman.tools.poisson_datagenerator import get_poisson_dataset
from hitman.neural_nets.poisson_chargenet import get_poisson_chargenet, poisson_nll_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help="Max epochs per model")
    parser.add_argument('--batch_power', type=int, default=15)
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    files = sorted(glob.glob("data/sbi_dataset_large/*.root"))
    Data = PoissonDataExtractor(files)
    charges, charge_hyp, pmt_positions, hit_obs, hit_hyp = Data.get_poisson_train_data()
    print(f"Data Loaded. Events: {len(charges)}, Sensors: {len(pmt_positions)}")

    hyp_norm_charge = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    obs_norm_charge = np.stack([np.std(pmt_positions, axis=0), np.mean(pmt_positions, axis=0)])
    hyp_norm_charge[0][hyp_norm_charge[0] == 0] = 1.0
    obs_norm_charge[0][obs_norm_charge[0] == 0] = 1.0

    N_events = len(charges)
    val_events = max(1, int(N_events * 0.1))
    train_events = N_events - val_events
    steps_train = int(train_events * len(pmt_positions) / (2**args.batch_power))
    steps_val = max(1, int(val_events * len(pmt_positions) / (2**args.batch_power)))

    print("Creating tf.data datasets...")
    train_gen = get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=2**args.batch_power, shuffle=True, split='train', val_fraction=0.1)
    val_gen = get_poisson_dataset(charges, charge_hyp, pmt_positions, batch_size=2**args.batch_power, shuffle=False, split='val', val_fraction=0.1)

    # 2. Define Grid
    layers_grid = [2, 3, 4]
    nodes_grid = [64, 128, 256]
    lr_grid = [0.005, 0.001, 0.0005]

    results = []

    out_dir = "networks_tune/poisson_grid_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(out_dir, exist_ok=True)
    
    strategy = tf.distribute.MirroredStrategy()

    for layers, nodes, lr in itertools.product(layers_grid, nodes_grid, lr_grid):
        name = f"L{layers}_N{nodes}_LR{lr}"
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        
        with strategy.scope():
            model = get_poisson_chargenet(layers=layers, nodes=nodes, hyp_norm=hyp_norm_charge, obs_norm=obs_norm_charge)
            model.layers[-1].activation = tf.keras.activations.linear
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss=poisson_nll_loss, optimizer=optimizer)

        # Use early stopping to abort quickly if validation loss plateaus
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)
        
        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            callbacks=[early_stopping, reduce_lr],
            verbose=2
        )
        
        best_val_loss = min(hist.history['val_loss'])
        best_train_loss = min(hist.history['loss'])
        epochs_run = len(hist.history['loss'])
        
        print(f"--> {name} Best Val Loss: {best_val_loss:.4f} (Train Loss: {best_train_loss:.4f}) in {epochs_run} epochs")
        
        results.append({
            'name': name,
            'layers': layers,
            'nodes': nodes,
            'lr': lr,
            'best_val_loss': best_val_loss,
            'best_train_loss': best_train_loss,
            'epochs_run': epochs_run
        })
        
        model.layers[-1].activation = tf.math.exp
        tf.keras.models.save_model(model, os.path.join(out_dir, name), save_format='tf')

    # Sort results by validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    summary_path = os.path.join(out_dir, "grid_search_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Poisson ChargeNet Grid Search Results\n")
        f.write("=====================================\n\n")
        f.write(f"{'Model':<20} | {'Val Loss':<10} | {'Train Loss':<10} | {'Epochs'}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['name']:<20} | {r['best_val_loss']:<10.4f} | {r['best_train_loss']:<10.4f} | {r['epochs_run']}\n")
            
    print(f"\nGrid search complete. Results saved to {summary_path}")
    print("Top 3 models:")
    for i in range(min(3, len(results))):
        print(f"{i+1}. {results[i]['name']} (Val Loss: {results[i]['best_val_loss']:.4f})")

if __name__ == "__main__":
    main()
