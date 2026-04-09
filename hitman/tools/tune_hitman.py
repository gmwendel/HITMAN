#!/usr/bin/env python3
import os
import subprocess
import itertools
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input files glob, e.g. ../../data/sbi_dataset_large/*.root', required=True)
    parser.add_argument('--epochs', default=50, type=int, help='Max epochs for tuning')
    args = parser.parse_args()

    input_files = args.input_files
    
    layers_list = [3, 4]
    nodes_list = [256, 512]
    batch_power_hitnet = [15, 16] # e.g. 2^15 or 2^16
    lr_list = [0.001]
    
    combinations = list(itertools.product(layers_list, nodes_list, batch_power_hitnet, lr_list))
    
    print(f"Starting Hyperparameter Tuning over {len(combinations)} configurations...")
    
    for (layers, nodes, bp_hit, lr) in combinations:
        bp_charge = bp_hit - 4 # Keep ChargeNet batch size scaled relatively to avoid zero-length validation batches
        
        output_dir = f"networks_tune/L{layers}_N{nodes}_B{bp_hit}_LR{lr}"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python", "-m", "hitman.train_hitman",
            "-i"
        ] + [input_files] + [
            "-o", output_dir,
            "--epochs", str(args.epochs),
            "--layers", str(layers),
            "--nodes", str(nodes),
            "--batch_power_hitnet", str(bp_hit),
            "--batch_power_chargenet", str(bp_charge),
            "--lr", str(lr)
        ]
        
        print(f"Running config: Layers={layers}, Nodes={nodes}, BatchPow={bp_hit}, LR={lr}")
        # Use shell=True for wildcard expansion in input_files
        cmd_str = " ".join(cmd)
        result = subprocess.run(cmd_str, shell=True)
        
        if result.returncode != 0:
            print(f"Configuration {output_dir} failed with return code {result.returncode}")

if __name__ == '__main__':
    main()
