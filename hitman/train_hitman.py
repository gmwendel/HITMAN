def main():
    args = get_args()
    from hitman.tools.ratextract import DataExtractor
    import numpy as np
    import glob
    
    expanded_files = []
    for f in args.input_files:
        expanded_files.extend(glob.glob(f))
    expanded_files = sorted(expanded_files)
    
    # load data
    Data = DataExtractor(expanded_files)
    charge_obs, hit_obs, charge_hyp, hit_hyp = Data.get_hitman_train_data()
    
    # Calculate norms dynamically
    charge_obs_norm = np.stack([np.std(charge_obs, axis=0), np.mean(charge_obs, axis=0)])
    charge_hyp_norm = np.stack([np.std(charge_hyp, axis=0), np.mean(charge_hyp, axis=0)])
    hit_obs_norm = np.stack([np.std(hit_obs, axis=0), np.mean(hit_obs, axis=0)])
    hit_hyp_norm = np.stack([np.std(hit_hyp, axis=0), np.mean(hit_hyp, axis=0)])
    
    # Prevent division by zero for constants
    charge_obs_norm[0][charge_obs_norm[0] == 0] = 1.0
    charge_hyp_norm[0][charge_hyp_norm[0] == 0] = 1.0
    hit_obs_norm[0][hit_obs_norm[0] == 0] = 1.0
    hit_hyp_norm[0][hit_hyp_norm[0] == 0] = 1.0

    print("Data Loaded")
    train_hitnet(args, hit_obs, hit_hyp, hit_hyp_norm, hit_obs_norm)
    train_chargenet(args, charge_obs, charge_hyp, charge_hyp_norm, charge_obs_norm)


def train_hitnet(args, hit_obs, hit_hyp, hyp_norm, obs_norm):
    import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import math
    from hitman.neural_nets.hitnet import get_hitnet, hitnet_trafo
    from hitman.tools.datagenerator import get_hitnet_dataset

    strategy = tf.distribute.MirroredStrategy()
    n_gpus = strategy.num_replicas_in_sync
    print("Number of devices: {}".format(n_gpus))
    optimizer = tf.keras.optimizers.Adam(args.lr)

    # Scale batch size with number of GPUs
    if n_gpus > 0:
        batch_scale = int(math.log(n_gpus, 2))
    else:
        batch_scale = 0
        
    N_hits = len(hit_obs)
    val_hits_num = max(1, int(N_hits * 0.1))
    train_hits_num = N_hits - val_hits_num
    
    half_batch_h = (2**(args.batch_power_hitnet + batch_scale)) // 2
    steps_train_h = int(train_hits_num / half_batch_h)
    steps_val_h = max(1, int(val_hits_num / half_batch_h))

    # Generate Training and Validation Datasets natively
    Train_Data = get_hitnet_dataset(hit_obs, hit_hyp, batch_size=2**(args.batch_power_hitnet + batch_scale), shuffle='inDOM', time_spread=args.t_shuffle, split='train', val_fraction=0.1)
    Val_Data = get_hitnet_dataset(hit_obs, hit_hyp, batch_size=2**(args.batch_power_hitnet + batch_scale), shuffle='inDOM', time_spread=args.t_shuffle, split='val', val_fraction=0.1)

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        if args.use_relu:
            hitnet = get_hitnet(activation='relu', layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        else:
            hitnet = get_hitnet(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        hitnet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], jit_compile=False)

    train_id = 'HITNET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

    #   Automatically train until validation loss does not decrease for 50 epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]
    #   additional callbacks for saving network as a function of epoch and extra analytics
    if args.save_history:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                args.output_network[0] + '/resources/checkpoints_hitnet/' + 'hitnet_{epoch:02d}',
                save_freq='epoch'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.output_network[0] + '/resources/logs_hitnet',
                                                        histogram_freq=1))

    hist = hitnet.fit(x=Train_Data,
                      validation_data=Val_Data,
                      epochs=int(args.epochs),
                      steps_per_epoch=steps_train_h,
                      validation_steps=steps_val_h,
                      verbose=2,
                      callbacks=callbacks)

    # save the trained network
    tf.keras.models.save_model(hitnet, args.output_network[0] + '/hitnet', save_format='tf')

    # load the network in without compiling and save with a linear activation required to get the LLH from the network
    # It seems this is a known issue:
    # https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py#L95

    linear_hitnet = tf.keras.models.load_model(args.output_network[0] + '/hitnet')
    linear_hitnet.layers[-1].activation = tf.keras.activations.linear
    tf.keras.models.save_model(linear_hitnet, args.output_network[0] + '/hitnet', save_format='tf')

    # summarize history for loss and accuracy
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Hitnet Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.output_network[0] + '/hitnet.png', dpi=200)
    plt.close()


def train_chargenet(args, charge_obs, charge_hyp, hyp_norm, obs_norm):
    import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import math
    from hitman.neural_nets.chargenet import get_chargenet, chargenet_trafo
    from hitman.tools.datagenerator import get_chargenet_dataset

    strategy = tf.distribute.MirroredStrategy()
    n_gpus = strategy.num_replicas_in_sync
    print("Number of devices: {}".format(n_gpus))
    optimizer = tf.keras.optimizers.Adam(args.lr * 0.1)

    # Scale batch size with number of GPUs
    if n_gpus > 1:
        batch_scale = int(math.log(n_gpus, 2))
    else:
        batch_scale = 0
        
    N_events = len(charge_obs)
    val_events_num = max(1, int(N_events * 0.1))
    train_events_num = N_events - val_events_num
    
    half_batch_c = (2**(args.batch_power_chargenet + batch_scale)) // 2
    steps_train_c = int(train_events_num / half_batch_c)
    steps_val_c = max(1, int(val_events_num / half_batch_c))

    # Generate Training and Validation Datasets natively on tf.data graph
    Train_Data = get_chargenet_dataset(charge_obs, charge_hyp, batch_size=2**(args.batch_power_chargenet + batch_scale), split='train', val_fraction=0.1)
    Val_Data = get_chargenet_dataset(charge_obs, charge_hyp, batch_size=2**(args.batch_power_chargenet + batch_scale), split='val', val_fraction=0.1)

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        if args.use_relu:
            chargenet = get_chargenet(activation='relu', layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        else:
            chargenet = get_chargenet(layers=args.layers, nodes=args.nodes, hyp_norm=hyp_norm, obs_norm=obs_norm)
        chargenet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], jit_compile=False)

    train_id = 'CHARGENET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

    #   Automatically train until validation loss does not decrease for 75 epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75)]
    #   additional callbacks for saving network as a function of epoch and extra analytics
    if args.save_history:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                args.output_network[0] + '/resources/checkpoints_chargenet/' + 'chargenet_{epoch:02d}',
                save_freq='epoch'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.output_network[0] + '/resources/logs_chargenet',
                                                        histogram_freq=1))

    hist = chargenet.fit(x=Train_Data,
                         validation_data=Val_Data,
                         epochs=int(args.epochs),
                         steps_per_epoch=steps_train_c,
                         validation_steps=steps_val_c,
                         verbose=2,
                         callbacks=callbacks)

    # save the trained network
    tf.keras.models.save_model(chargenet, args.output_network[0] + '/chargenet', save_format='tf')

    # load the network in without compiling and save with a linear activation required to get the LLH from the network
    # It seems this is a known issue:
    # https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py#L95

    linear_chargenet = tf.keras.models.load_model(args.output_network[0] + '/chargenet')
    linear_chargenet.layers[-1].activation = tf.keras.activations.linear
    tf.keras.models.save_model(linear_chargenet, args.output_network[0] + '/chargenet', save_format='tf')

    # summarize history for loss and accuracy
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Chargenet Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.output_network[0] + '/chargenet.png', dpi=200)
    plt.close()


def get_args():
    import argparse
    # Get command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.root', nargs='+',
                        required=True)
    parser.add_argument('-o', '--output_network',
                        help='Type = String; Output location for trained network, e.g. networks/', nargs=1,
                        required=True)
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Type = Integer. Optional; limit number of epochs; Default = 1000')
    parser.add_argument('--t_shuffle', default=50, type=int,
                        help='Type = Integer. Optional; Sets the standard deviation of the time shuffling; Default = 75')
    parser.add_argument('--use_relu', default=False, action="store_true",
                        help='Type = Boolean.  Optional; Use the relu activation function instead of mish; Default = False')
    parser.add_argument('--save_history', default=False, action="store_true",
                        help="Type = Boolean.  Optional; Add flag to save network at each epoch and enable Tensorboard stats in 'resource' folder ; Default = False")
    parser.add_argument('--layers', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('--nodes', default=256, type=int, help='Nodes per hidden layer')
    parser.add_argument('--batch_power_hitnet', default=17, type=int, help='Base 2 power for HitNet batch size')
    parser.add_argument('--batch_power_chargenet', default=14, type=int, help='Base 2 power for ChargeNet batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    main()
