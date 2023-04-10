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
    parser.add_argument('--event_start', default=0, type=int,
                        help='Type = Integer. Optional; Sets the start number of events to reconstruct; Default = all events',
                        required=False)

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
    def uniform_sample(samples, half_z, half_r, t_min, t_max, E_min, E_max):
        length = np.random.uniform(0, 1, size=(samples, 1))
        angle = np.pi * np.random.uniform(0, 2, size=(samples, 1))

        x = half_r * 0.9 * np.sqrt(length) * np.cos(angle)
        y = half_r * 0.9 * np.sqrt(length) * np.sin(angle)
        z = np.random.uniform(-half_z * 0.9, half_z * 0.9, size=(samples, 1))

        # Not Properly distribute points on surface of sphere
        zenith = np.arccos(np.random.uniform(-1, 1, size=(samples, 1)))
        azimuth = np.random.uniform(0, 2 * np.pi, size=(samples, 1))

        t = np.random.uniform(t_min, t_max, size=(samples, 1))
        E = np.random.uniform(E_min, E_max, size=(samples, 1))
        # stack initial points
        initial_points = np.hstack([x, y, z, zenith, azimuth, t, E]).astype(np.float32)
        return initial_points

    # Use random grid sampling to find best -LLH values before gradient descent
    def best_guess(hitnet, chargenet, event, final_number, samples, half_z, half_r, t_min, t_max, E_min, E_max):
        all_points = uniform_sample(samples, half_z, half_r, t_min, t_max, E_min, E_max)
        all_llh = tfLLH(event['hits'], all_points, hitnet, event['total_charge'], chargenet).numpy()
        for i in range(10):
            initial_points = uniform_sample(samples, half_z, half_r, t_min, t_max, E_min, E_max)
            llh = tfLLH(event['hits'], initial_points, hitnet, event['total_charge'], chargenet).numpy()
            all_points = np.vstack([all_points, initial_points])
            all_llh = np.vstack([all_llh, llh])
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
        #    print(theta)
        out = tf.reshape(NLLH, (hits.shape[0], theta.shape[0]))
        out = tf.math.reduce_sum(out, axis=0)
        out = out - tf.transpose(chargenet([c, theta]))
        return out[0]

    # Spherical coordinates are cyclic, fix going beyond bounds.  e.g. azimuth 3pi = pi
    def proper_dir(zenith, azimuth):

        u = np.sin(zenith) * np.cos(azimuth)
        v = np.sin(zenith) * np.sin(azimuth)
        w = np.cos(zenith)
        az = np.arctan2(v, u)
        az = np.less(az, 0) * 2 * np.pi + az
        ze = np.arccos(w)

        return ze, az

    # Where the magic happens, gradient descent optimizer, need option for wbls vs water since step size changes
    def eval_with_grads(hits, params, hitnet, charge, chargenet, printall=False):
        all_llhs = []
        all_params = []
        params = tf.convert_to_tensor(params, np.float32)

        descent_rates = tf.tile([[400., 400., 400., 0.1, 0.1, 0.013, 0.006]], (len(params), 1)) * 95 / (
                len(hits) + 7) * 0.1  # wbls best

        # descent_rates=tf.tile([[600.,600.,600.,0.1,0.1,0.00013,0.1]],(len(params),1))*160/(len(hits)+15)*0.1 #gentle t
        # descent_rates = tf.tile([[800., 800., 800., 0.01, 0.01, 0.013, 0.003]], (len(params), 1)) * 0.25 * 95 / (
        # len(hits) + 15)  # water best

        for i in range(0, 250):
            with tf.GradientTape() as g:
                g.watch(params)
                llhs = tfLLH(hits, params, hitnet, charge, chargenet)

            grads = g.gradient(llhs, params)

            all_llhs.append(llhs.numpy())
            all_params.append(params.numpy())
            params = params - descent_rates * grads  # relu

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
    events = events[args.event_start:args.event_limit]
    print('number of events to reconstruct: ', len(events))

    samples = 1500  # specifies batch size for initial grid search
    final_number = 150  # specifies batch size for gradient descent
    i = 0

    # Optimize over all events loaded
    for event in events:
        # generate 'best guess'
        initial_points = best_guess(hitnet, chargenet, event, final_number, samples, float(args.half_height),
                                    float(args.radius), -5, 5, 1.25, 2.75)
        event_results = eval_with_grads(event['hits'], initial_points, hitnet, event['total_charge'], chargenet)
        llhmin = np.min(event_results[2])
        llh = event_results[0].numpy()
        index = np.where(llh == llhmin)
        a, b = np.where(event_results[2] == np.min(event_results[2]))
        print(llhmin)

        # Add reco to file

        event['reco'] = event_results[3][a[0]][b[0]]
        event['reco'][3:5] = proper_dir(event['reco'][3], event['reco'][4])
        event['reco_LLH'] = llhmin

        print('reconstruction finished for event #' + str(i))
        print('event results: ', event['reco'])
        i = i + 1

    # Save file with reconstructions
    fileObj = open(args.output_file, 'wb')
    pickle.dump(events, fileObj)
    fileObj.close()
    exit()
