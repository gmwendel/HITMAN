#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import pickle
import glob
import os
from hitman.tools.ratextract_poisson import PoissonDataExtractor
from hitman.neural_nets.poisson_chargenet import poisson_chargenet_trafo

@tf.function
def tfLLH_joint_poisson(joint_charges, N_events, pmt_positions, theta_batch, model):
    N_sensors = tf.shape(pmt_positions)[0]
    batch_size = tf.shape(theta_batch)[0]
    
    # Tile PMT positions for every hypothesis in the batch
    h = tf.tile(pmt_positions, (batch_size, 1))
    
    # Repeat each hypothesis N_sensors times
    p = tf.repeat(theta_batch, N_sensors, axis=0)
    
    # Predict log-lambda (z)
    z_pred = model([h, p])
    z_pred = tf.clip_by_value(z_pred, -20.0, 20.0)
    
    # Reshape back to (batch_size, N_sensors)
    z_pred_reshaped = tf.reshape(z_pred, (batch_size, N_sensors))
    
    # Tile joint_charges for each hypothesis in the batch
    c_total = tf.tile(joint_charges, (batch_size,))
    c_total = tf.cast(c_total, tf.float32)
    c_total = tf.reshape(c_total, (batch_size, N_sensors))
    
    # Calculate Joint Poisson NLL: N * e^z - (sum k) * z
    nll = tf.cast(N_events, tf.float32) * tf.math.exp(z_pred_reshaped) - c_total * z_pred_reshaped
    
    # Sum over all sensors
    total_nll = tf.math.reduce_sum(nll, axis=1)
    
    return total_nll

def iterative_random_search(model, joint_event, samples_per_stage, stages, zoom_factor, vram_batch_size):
    abs_bounds = np.array([
        [0.1, 0.6],       # Energy
        [0.2, 1.0],       # Scat
        [2000.0, 15000.0] # Abs
    ])
    
    current_bounds = np.copy(abs_bounds)
    best_point = None
    best_llh = np.inf
    
    joint_charges = joint_event['charges']
    N_events = joint_event['N_events']
    pmt_positions = joint_event['pmt_positions']
    
    N_sensors = len(pmt_positions)
    # Safe batch size to prevent OOM
    actual_batch_size = max(1, vram_batch_size // N_sensors)
    
    for stage in range(stages):
        energy = np.random.uniform(current_bounds[0, 0], current_bounds[0, 1], size=(samples_per_stage, 1))
        scat = np.random.uniform(current_bounds[1, 0], current_bounds[1, 1], size=(samples_per_stage, 1))
        abs_len = np.random.uniform(current_bounds[2, 0], current_bounds[2, 1], size=(samples_per_stage, 1))
        points = np.hstack([energy, scat, abs_len]).astype(np.float32)
        
        llhs = []
        for i in range(0, samples_per_stage, actual_batch_size):
            batch_points = points[i:i+actual_batch_size]
            batch_llhs = tfLLH_joint_poisson(joint_charges, N_events, pmt_positions, batch_points, model).numpy()
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
    parser.add_argument('--vram_batch_size', default=1000000, type=int)
    parser.add_argument('--events_per_group', default=10, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    
    expanded_files = []
    for f in args.input_files:
        expanded_files.extend(glob.glob(f))
        
    expanded_files = sorted(expanded_files)
        
    Data = PoissonDataExtractor(expanded_files)
    events = Data.get_poisson_reco_data()
    print(f'Data loaded. Number of raw events: {len(events)}')
    
    grouped_events = []
    group_size = args.events_per_group
    for i in range(0, len(events), group_size):
        chunk = events[i:i+group_size]
        joint_charges = np.sum([e['charges'] for e in chunk], axis=0)
        grouped_events.append({
            'charges': joint_charges,
            'N_events': len(chunk),
            'pmt_positions': chunk[0]['pmt_positions'],
            'truth': chunk[0]['truth'] # They all share the same truth
        })
    print(f'Grouped into {len(grouped_events)} joint events of size {group_size}')
    
    model = tf.keras.models.load_model(
        args.network + '/poisson_chargenet', 
        custom_objects={'poisson_chargenet_trafo': poisson_chargenet_trafo},
        compile=False
    )
    
    for i, event in enumerate(grouped_events):
        llhmin, best_point = iterative_random_search(
            model=model,
            joint_event=event,
            samples_per_stage=args.samples_per_stage,
            stages=args.stages,
            zoom_factor=args.zoom_factor,
            vram_batch_size=args.vram_batch_size
        )
        
        event['reco'] = best_point
        event['reco_LLH'] = llhmin
        
        print(f'reconstruction finished for joint event #{i}')
        print(f'event results: {event["reco"]}')
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(grouped_events, f)

if __name__ == '__main__':
    main()
