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
    parser.add_argument('-r', '--radius',
                        help='Type = int;  Specify detector radius in mm',
                        nargs=None,
                        required=True
                        )
    parser.add_argument('-z', '--half_height',
                        help='Type = int;  Specify detector half-height in mm',
                        nargs=None,
                        required=True
                        )
    parser.add_argument('--event_limit', default=-1, type=int,
                        help='Type = Integer. Optional; Sets the max number of events to reconstruct; Default = all events',
                        required=False)

    parser.add_argument('--print_numpy', default=False, type=bool,
                        help='Type = Boolean;  Prints additional numpy files about failed events, etc.; Default = False'
                        )

    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    import pickle
    from hitman.tools.ratextract import DataExtractor

    # Generate uniform space to seed optimizer
    def uniform_sample(samples, e_min=0.1, e_max=0.6, scat_min=0.2, scat_max=1.0, abs_min=2000.0, abs_max=15000.0):
        energy = np.random.uniform(e_min, e_max, size=(samples, 1))
        scat = np.random.uniform(scat_min, scat_max, size=(samples, 1))
        abs_len = np.random.uniform(abs_min, abs_max, size=(samples, 1))
        initial_points = np.hstack([energy, scat, abs_len]).astype(np.float32)
        return initial_points

    # Use random grid sampling to find best -LLH values before gradient descent
    def best_guess(hitnet, chargenet, event, final_number, samples):
        all_points = uniform_sample(samples)
        all_llh = tfLLH(event['hits'], all_points, hitnet, event['total_charge'], chargenet).numpy()
        for i in range(20):
            initial_points = uniform_sample(samples)
            llh = tfLLH(event['hits'], initial_points, hitnet, event['total_charge'], chargenet).numpy()
            all_points = np.vstack([all_points, initial_points])
            all_llh = np.hstack([all_llh, llh])
        n_minLLH = np.argpartition(all_llh, final_number)
        return all_points[n_minLLH[:final_number], :]

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

    # Where the magic happens, gradient descent optimizer
    def eval_with_grads(hits, params, hitnet, charge, chargenet, printall=False):
        all_llhs = []
        all_params = []
        params = tf.convert_to_tensor(params, np.float32)

        # Descent rates tuned roughly for the optical parameters [Energy, Scat, Abs]
        descent_rates = tf.tile([[0.005, 0.005, 100.0]], (len(params), 1)) * 95 / (len(hits) + 7) * 0.1

        for i in range(0, 250):
            with tf.GradientTape() as g:
                g.watch(params)
                llhs = tfLLH(hits, params, hitnet, charge, chargenet)

            grads = g.gradient(llhs, params)

            all_llhs.append(llhs.numpy())
            all_params.append(params.numpy())
            params = params - descent_rates * grads

        return llhs, params, all_llhs, all_params

    def calc_n9(event, theta):
        x = [theta[0], theta[1], theta[2], theta[5]]
        c = 299792458 * 10 ** -6  # mm/ns
        n = 1.333
        x = theta
        hit = event['hits']
        residuals = (hit[:, 3] - x[5]) - n / c * (
                (x[0] - hit[:, 0]) ** 2 + (x[1] - hit[:, 1]) ** 2 + (x[2] - hit[:, 2]) ** 2) ** 0.5
        lower = -3
        upper = 6
        out = np.where((residuals > lower) & (residuals < upper))
        return len(out[0])

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

    samples = 1000  # specifies batch size for initial grid search
    final_number = 150  # specifies batch size for gradient descent
    i = 0

    # Optimize over all events loaded
    for event in events:
        # generate 'best guess'
        initial_points = best_guess(hitnet, chargenet, event, final_number, samples)
        event_results = eval_with_grads(event['hits'], initial_points, hitnet, event['total_charge'], chargenet)
        llhmin = np.min(event_results[2])
        llh = event_results[0].numpy()
        index = np.where(llh == llhmin)
        a, b = np.where(event_results[2] == np.min(event_results[2]))
        print(llhmin)

        # Add reco to file

        event['reco'] = event_results[3][a[0]][b[0]]
        event['reco_LLH'] = llhmin

        print('reconstruction finished for event #' + str(i))
        print('event results: ', event['reco'])
        i = i + 1

    # Save file with reconstructions
    fileObj = open(args.output_file, 'wb')
    pickle.dump(events, fileObj)
    fileObj.close()
    exit()
