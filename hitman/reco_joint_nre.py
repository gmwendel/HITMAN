#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import pickle
import glob
import os
from hitman.tools.ratextract import DataExtractor

@tf.function
def tfLLH_joint(all_hits, all_charges, theta_batch, hitnet, chargenet):
    num_params = tf.shape(theta_batch)[0]
    
    # Hit LLH
    h = tf.repeat(all_hits, num_params, axis=0)
    p_hit = tf.tile(theta_batch, (tf.shape(all_hits)[0], 1))
    NLLH_hit = -hitnet([h, p_hit])
    out_hit = tf.reshape(NLLH_hit, (tf.shape(all_hits)[0], num_params))
    total_hit_llh = tf.math.reduce_sum(out_hit, axis=0)
    
    # Charge LLH
    c = tf.repeat(all_charges, num_params, axis=0)
    p_charge = tf.tile(theta_batch, (tf.shape(all_charges)[0], 1))
    LLH_charge = chargenet([c, p_charge])
    out_charge = tf.reshape(LLH_charge, (tf.shape(all_charges)[0], num_params))
    total_charge_llh_to_subtract = tf.math.reduce_sum(out_charge, axis=0)
    
    return total_hit_llh - total_charge_llh_to_subtract

def iterative_random_search(hitnet, chargenet, all_hits, all_charges, samples_per_stage, stages, zoom_factor, vram_batch_size):
    abs_bounds = np.array([
        [0.1, 0.6],       # Energy
        [0.2, 1.0],       # Scat
        [2000.0, 15000.0] # Abs
    ])
    
    current_bounds = np.copy(abs_bounds)
    best_point = None
    best_llh = np.inf
    
    num_hits = len(all_hits)
    actual_batch_size = max(1, vram_batch_size // max(num_hits, 1))
    
    for stage in range(stages):
        energy = np.random.uniform(current_bounds[0, 0], current_bounds[0, 1], size=(samples_per_stage, 1))
        scat = np.random.uniform(current_bounds[1, 0], current_bounds[1, 1], size=(samples_per_stage, 1))
        abs_len = np.random.uniform(current_bounds[2, 0], current_bounds[2, 1], size=(samples_per_stage, 1))
        points = np.hstack([energy, scat, abs_len]).astype(np.float32)
        
        llhs = []
        for i in range(0, samples_per_stage, actual_batch_size):
            batch_points = points[i:i+actual_batch_size]
            batch_llhs = tfLLH_joint(all_hits, all_charges, batch_points, hitnet, chargenet).numpy()
            llhs.append(batch_llhs)
            
        all_llhs = np.concatenate(llhs)
        
        min_idx = np.argmin(all_llhs)
        stage_best_llh = all_llhs[min_idx]
        stage_best_point = points[min_idx]
        
        if stage_best_llh < best_llh:
            best_llh = stage_best_llh
            best_point = stage_best_point
            
        current_width = current_bounds[:, 1] - current_bounds[:, 0]
        new_half_width = (current_width * zoom_factor) / 2
        
        current_bounds[:, 0] = np.clip(best_point - new_half_width, abs_bounds[:, 0], abs_bounds[:, 1])
        current_bounds[:, 1] = np.clip(best_point + new_half_width, abs_bounds[:, 0], abs_bounds[:, 1])
        
    return best_llh, best_point

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input locations', nargs='+', required=True)
    parser.add_argument('-n', '--network', help='Location of trained network', required=True)
    parser.add_argument('-o', '--output_file', help='Output file', required=True)
    parser.add_argument('--samples_per_stage', default=100000, type=int)
    parser.add_argument('--stages', default=5, type=int)
    parser.add_argument('--zoom_factor', default=0.1, type=float)
    parser.add_argument('--vram_batch_size', default=10000000, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    
    expanded_files = []
    for f in args.input_files:
        expanded_files.extend(glob.glob(f))
    expanded_files = sorted(expanded_files)
    
    Data = DataExtractor(expanded_files)
    events = Data.get_hitman_reco_data()
    print(f'Data loaded. Number of events to reconstruct: {len(events)}')
    
    hitnet = tf.keras.models.load_model(args.network + '/hitnet')
    hitnet.layers[-1].activation = tf.keras.activations.linear
    chargenet = tf.keras.models.load_model(args.network + '/chargenet')
    chargenet.layers[-1].activation = tf.keras.activations.linear
    
    # Aggregate all hits and charges
    all_hits = np.concatenate([e['hits'] for e in events], axis=0)
    all_charges = np.stack([e['total_charge'] for e in events], axis=0)
    print(f'Aggregated {len(all_hits)} hits and {len(all_charges)} charges.')
    
    llhmin, best_point = iterative_random_search(
        hitnet=hitnet, 
        chargenet=chargenet, 
        all_hits=all_hits, 
        all_charges=all_charges, 
        samples_per_stage=args.samples_per_stage, 
        stages=args.stages, 
        zoom_factor=args.zoom_factor, 
        vram_batch_size=args.vram_batch_size
    )
    
    result = [{
        'reco': best_point,
        'reco_LLH': llhmin,
        'N_events_stacked': len(events),
        'truth': events[0]['truth'] # all events share same truth
    }]
    
    print(f'Reconstruction finished for all {len(events)} stacked events')
    print(f'Event results: {best_point}')
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()