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

    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pickle
    from hitman.tools.CCextract import DataExtractor

    # Generate uniform space to seed optimizer
    def uniform_sample(samples, t_min, t_max, E_min, E_max):
        length = np.random.uniform(0, 1, size=(samples, 1))
        angle = np.pi * np.random.uniform(0, 2, size=(samples, 1))

        # Not Properly distribute points on surface of sphere
        zenith = np.arccos(np.random.uniform(-1, 1, size=(samples, 1)))
        azimuth = np.random.uniform(0, 2 * np.pi, size=(samples, 1))

        t = np.random.uniform(t_min, t_max, size=(samples, 1))
        E = np.random.uniform(E_min, E_max, size=(samples, 1))
        # stack initial points
        initial_points = np.hstack([zenith, azimuth, E, t]).astype(np.float32)
        return initial_points

    # Use random grid sampling to find best -LLH values before gradient descent
    def best_guess(hitnet, chargenet, event, final_number, samples, t_min, t_max, E_min, E_max):
        all_points = uniform_sample(samples, t_min, t_max, E_min, E_max)
        all_llh = tfLLH(event['hits'], all_points, hitnet, event['total_charge'], chargenet).numpy()
        for i in range(20):
            initial_points = uniform_sample(samples, t_min, t_max, E_min, E_max)
            llh = tfLLH(event['hits'], initial_points, hitnet, event['total_charge'], chargenet).numpy()
            all_points = np.vstack([all_points, initial_points])
        n_minLLH = np.argpartition(all_llh, final_number)
        return all_points[n_minLLH[:final_number], :]

    # Define function that evaluates the negative log-likelihood
    #    @tf.function
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

    def LLH(hits, theta, hitnet, charge, chargenet):
        return tfLLH(hits, theta, hitnet, charge, chargenet).numpy()

    def safe_LLH(hits, theta, hitnet, charge, chargenet):
        out = []
        split_theta = np.array_split(theta, 1 + int(len(theta) / 3000))
        for t in split_theta:
            out.append(LLH(hits, t, hitnet, charge, chargenet))
        return np.concatenate(out)

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

    def corner_plot(event):
        scale = 1
        font = 36 * scale
        plt.rcParams.update({'font.size': font})
        pos_size = 10
        t_size = .5
        resolution = 100
        reco = event['truth']

        print(reco)
        x = np.linspace(-pos_size, pos_size, resolution)
        y = np.linspace(-pos_size, pos_size, resolution)
        z = np.linspace(-pos_size, pos_size, resolution)
        ze = np.linspace(0, np.pi, resolution)
        az = np.linspace(0, 2 * np.pi, resolution)
        t = np.linspace(-t_size, t_size, resolution)
        E = np.linspace(-.5, .5, resolution)
        dimensions = [x, y, z, ze, az, t, E]

        name = ['x (mm)', 'y (mm)', 'z (mm)', 'ze (rad)', 'az (rad)', 't (ns)', 'E (MeV)']
        dimensions = [x, y, z, ze, az, t, E]
        dims = 7
        fig, ax = plt.subplots(dims, dims, figsize=(32 * scale, 32 * scale))
        for i in range(0, dims):
            for j in range(0, i):
                print(i, j)
                d1, d2 = np.meshgrid(dimensions[j], dimensions[i])
                n_evals = d1.flatten() * resolution ** 2
                base = np.zeros((resolution ** 2, 7))
                base[:, j] = base[:, j] + d1.flatten()
                base[:, i] = base[:, i] + d2.flatten()

                theta = base + np.tile(event['truth'], (resolution ** 2, 1))

                if j == 3 or j == 4:
                    theta[:, j] = d1.flatten()
                if i == 3 or i == 4:
                    theta[:, i] = d2.flatten()

                scan = safe_LLH(event['hits'], theta, hitnet, event['total_charge'], chargenet)
                scan = np.reshape(scan, (-1, resolution))

                if j == 3 or j == 4:
                    xdim = dimensions[j]
                else:
                    xdim = dimensions[j] + reco[j]
                if i == 3 or i == 4:
                    ydim = dimensions[i]
                else:
                    ydim = dimensions[i] + reco[i]
                ax[i, j].pcolormesh(xdim, ydim, scan)
                ax[i, j].axhline(reco[i], color='r')
                ax[i, j].axvline(reco[j], color='r')
                ax[i, j].scatter(event['truth'][j], event['truth'][i], marker='*', s=font * 4, color='white')
                # plt.colorbar(label="delta -LLH")
                ax[i, j].set(xlabel=name[j], ylabel=name[i])
                if (i > 0 and i < dims - 1):
                    ax[i, j].xaxis.set_visible(False)
                if (j > 0):
                    ax[i, j].yaxis.set_visible(False)
        twoDscan = scan
        for i in range(0, dims):

            theta = np.tile(event['truth'], (resolution, 1))

            if i == 0 or i == 1:
                theta[:, i] = dimensions[i]
            else:
                theta[:, i] = dimensions[i] + theta[:, i]

            scan = safe_LLH(event['hits'], theta, hitnet, event['total_charge'], chargenet)

            if i == 3 or i == 4:
                xdim = dimensions[i]
            else:
                xdim = dimensions[i] + reco[i]

            ax[i, i].plot(xdim, scan)
            # plt.colorbar(label="delta -LLH")
            ax[i, i].set(xlabel=name[i])
            ax[i, i].axis('off')

        for i in range(0, dims):
            for j in range(i + 1, dims):
                fig.delaxes(ax[i][j])
        #    fig.tight_layout()
        return plt, twoDscan

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

    all_llhs = []
    for i in range(400):
        print(len(events[i]['hits']))
        dimsize = 150  # specifies batch size for grid
        resolution = 150
        ze = np.linspace(0, np.pi, resolution)
        az = np.linspace(0, 2 * np.pi, resolution)

        plot, twoDscan = corner_plot(events[i])
        plot.savefig("plots/" + str(i).zfill(3) + '.png')
        plot.close()
        #        print(best_guess(hitnet, chargenet, events[i], final_number=10, samples=1000, t_min=events[i]['truth'][3], t_max=events[i]['truth'][3], E_min=events[i]['truth'][2], E_max=events[i]['truth'][2]))

        twoDscan = twoDscan + len(events[i]['hits']) + 1  # normalize for sum
        all_llhs.append(twoDscan)

    all_llhs = np.array(all_llhs).astype(np.float64)
    total = np.sum(all_llhs, axis=0)
    plt.figure(figsize=(32, 32))
    plt.pcolormesh(ze, az, total)
    plt.axhline(events[0]['truth'][4], color='r')
    plt.axvline(events[0]['truth'][3], color='r')
    plt.title('Summed LLH')
    plt.xlabel('zenith (rad)')
    plt.ylabel('azimuth (rad)')
    plt.savefig("plots/total.png")
    idx_ze, idx_az = np.where(total == np.min(total))
    print(ze[idx_ze], az[idx_az])


if __name__ == '__main__':
    main()
