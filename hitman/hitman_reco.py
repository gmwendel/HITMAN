import numpy as np
import tensorflow as tf
import pickle
from hitman.tools.ratextract import DataExtractor

# Define function that evaluates the negative log-likelihood
@tf.function
def tfLLH(hits, theta, hitnet, charge, chargenet):
    num_params = tf.shape(theta)[0]
    h = tf.repeat(hits, num_params, axis=0)
    p = tf.tile(theta, (hits.shape[0], 1))
    c = tf.repeat([charge], num_params, axis=0)
    NLLH = -hitnet([h, p])
    out = tf.reshape(NLLH, (hits.shape[0], theta.shape[0]))
    out = tf.math.reduce_sum(out, axis=0)
    out = out - tf.transpose(chargenet([c, theta]))
    return out[0]

def iterative_random_search(hitnet, chargenet, event, samples_per_stage, stages, zoom_factor, vram_batch_size):
    # Absolute physical bounds
    abs_bounds = np.array([
        [0.1, 0.6],       # Energy
        [0.2, 1.0],       # Scat
        [2000.0, 15000.0] # Abs
    ])

    current_bounds = np.copy(abs_bounds)
    best_point = None
    best_llh = np.inf

    # Calculate a safe chunk size based on the number of hits in this specific event
    # vram_batch_size now represents the maximum number of network inferences per chunk
    num_hits = len(event['hits'])
    actual_batch_size = max(1, vram_batch_size // num_hits)

    for stage in range(stages):        # Sample hypotheses within current bounds
        energy = np.random.uniform(current_bounds[0, 0], current_bounds[0, 1], size=(samples_per_stage, 1))
        scat = np.random.uniform(current_bounds[1, 0], current_bounds[1, 1], size=(samples_per_stage, 1))
        abs_len = np.random.uniform(current_bounds[2, 0], current_bounds[2, 1], size=(samples_per_stage, 1))
        points = np.hstack([energy, scat, abs_len]).astype(np.float32)
        
        # Evaluate in chunks
        llhs = []
        for i in range(0, samples_per_stage, actual_batch_size):
            batch_points = points[i:i+actual_batch_size]
            batch_llhs = tfLLH(event['hits'], batch_points, hitnet, event['total_charge'], chargenet).numpy()
            llhs.append(batch_llhs)
        
        all_llhs = np.concatenate(llhs)
        
        # Find best
        min_idx = np.argmin(all_llhs)
        stage_best_llh = all_llhs[min_idx]
        stage_best_point = points[min_idx]
        
        if stage_best_llh < best_llh:
            best_llh = stage_best_llh
            best_point = stage_best_point
            
        # Calculate new bounds for next stage
        current_width = current_bounds[:, 1] - current_bounds[:, 0]
        new_half_width = (current_width * zoom_factor) / 2
        
        current_bounds[:, 0] = np.clip(best_point - new_half_width, abs_bounds[:, 0], abs_bounds[:, 1])
        current_bounds[:, 1] = np.clip(best_point + new_half_width, abs_bounds[:, 0], abs_bounds[:, 1])
        
    return best_llh, best_point

def main():
    import argparse

    # Get command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; locations of events to be evaluated, e.g. $PWD/{1..16}.pkl',
                        nargs='+',
                        required=True
                        )
    parser.add_argument('-n', '--network',
                        help='Type = String; Location of trained network to be used for reconstruction, e.g. $PWD/networks',
                        nargs=None,
                        required=True
                        )
    parser.add_argument('-o', '--output_file',
                        help='Type = String;  Specify output file with reconstructed values; e.g. $PWD/reco.pkl',
                        nargs=None,
                        required=True
                        )
    parser.add_argument('--event_limit', default=-1, type=int,
                        help='Type = Integer. Optional; Sets the max number of events to reconstruct; Default = all events',
                        required=False)
    parser.add_argument('--print_numpy', default=False, type=bool,
                        help='Type = Boolean;  Prints additional numpy files about failed events, etc.; Default = False'
                        )
    parser.add_argument('--samples_per_stage', default=100000, type=int,
                        help='Type = Integer; Samples to evaluate per grid search stage; Default = 100000')
    parser.add_argument('--stages', default=5, type=int,
                        help='Type = Integer; Number of zoom stages; Default = 5')
    parser.add_argument('--zoom_factor', default=0.1, type=float,
                        help='Type = Float; Fraction of previous bounds to retain in zoom; Default = 0.1')
    parser.add_argument('--vram_batch_size', default=25000, type=int,
                        help='Type = Integer; Max samples to evaluate simultaneously to avoid OOM; Default = 25000')

    args = parser.parse_args()

    # load hitnet & chargenet
    hitnet = tf.keras.models.load_model(args.network + '/hitnet')
    hitnet.layers[-1].activation = tf.keras.activations.linear
    chargenet = tf.keras.models.load_model(args.network + '/chargenet')
    chargenet.layers[-1].activation = tf.keras.activations.linear

    # Load data for reconstruction
    Data = DataExtractor(args.input_files)
    events = Data.get_hitman_reco_data()
    print('data loaded')
    print(len(events))
    events = events[:args.event_limit]
    print('number of events to reconstruct: ', len(events))

    i = 0

    # Optimize over all events loaded
    for event in events:
        llhmin, best_point = iterative_random_search(
            hitnet=hitnet, 
            chargenet=chargenet, 
            event=event, 
            samples_per_stage=args.samples_per_stage, 
            stages=args.stages, 
            zoom_factor=args.zoom_factor, 
            vram_batch_size=args.vram_batch_size
        )

        # Add reco to file
        event['reco'] = best_point
        event['reco_LLH'] = llhmin

        print('reconstruction finished for event #' + str(i))
        print('event results: ', event['reco'])
        i = i + 1

    # Save file with reconstructions
    fileObj = open(args.output_file, 'wb')
    pickle.dump(events, fileObj)
    fileObj.close()
    exit()

if __name__ == '__main__':
    main()
