def main():
    args = get_args()
    from hitman.tools.CCextract import DataExtractor
    # load data
    Data = DataExtractor(args.input_files)
    charge_obs, hit_obs, charge_hyp, hit_hyp = Data.get_hitman_train_data()
    print("Data Loaded")
    train_hitnet(args, hit_obs, hit_hyp)
    train_chargenet(args, charge_obs, charge_hyp)


def train_hitnet(args, hit_obs, hit_hyp):
    import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import math
    from hitman.neural_nets.hitnet import get_hitnet
    from hitman.tools.datagenerator import DataGenerator

    strategy = tf.distribute.MirroredStrategy()
    n_gpus = strategy.num_replicas_in_sync
    print("Number of devices: {}".format(n_gpus))
    optimizer = tf.keras.optimizers.Adam(0.0001)

    # Take 1/10 total data and make it validation
    splits = int(len(hit_obs) / 10)
    # Scale batch size with number of GPUs
    if n_gpus > 0:
        batch_scale = int(math.log(n_gpus, 2))
    else:
        batch_scale = 0
    # Generate Training and Validation Datasets
    Train_Data = DataGenerator(hit_obs[0:-splits], hit_hyp[0:-splits], batch_size=2 ** (17 + batch_scale),
                               )
    Val_Data = DataGenerator(hit_obs[-splits:-1], hit_hyp[-splits:-1], batch_size=2 ** (17 + batch_scale),
                             )
    del hit_obs
    del hit_hyp

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        if args.use_relu:
            hitnet = get_hitnet(activation='relu')
        else:
            hitnet = get_hitnet()
        hitnet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], jit_compile=False)

    train_id = 'HITNET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

    #   Automatically train until validation loss does not decrease for 25 epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)]
    #   additional callbacks for saving network as a function of epoch and extra analytics
    if args.save_history:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(save_freq='epoch',
                                                            path_template=args.output_network[
                                                                              0] + 'resources/checkpoints/' + train_id + '{epoch:02d}'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.output_network[0] + 'resources/logs/' + train_id,
                                                        histogram_freq=1))

    hist = hitnet.fit(x=Train_Data,
                      validation_data=Val_Data,
                      epochs=int(args.epochs),
                      verbose=2,
                      callbacks=callbacks,
                      use_multiprocessing=True,
                      max_queue_size=512,
                      workers=2*n_gpus)

    # save the trained network
    tf.keras.models.save_model(hitnet, args.output_network[0] + '/hitnet', save_format='tf')
    tf.keras.models.save_model(hitnet, args.output_network[0] + '/hitnet.h5',
                               save_format='h5')  # Old format (HITMAN 0.1)

    # summarize history for loss and accuracy
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Hitnet Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.output_network[0] + '/hitnet.png', dpi=200)


def train_chargenet(args, charge_obs, charge_hyp):
    import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import math
    from hitman.neural_nets.chargenet import get_chargenet
    from hitman.tools.datagenerator import DataGenerator

    strategy = tf.distribute.MirroredStrategy()
    n_gpus = strategy.num_replicas_in_sync
    print("Number of devices: {}".format(n_gpus))
    optimizer = tf.keras.optimizers.Adam(0.001)

    # Take 1/10 total data and make it validation
    splits = int(len(charge_obs) / 10)
    # Scale batch size with number of GPUs
    if n_gpus > 1:
        batch_scale = int(math.log(n_gpus, 2))
    else:
        batch_scale = 0
    # Generate Training and Validation Datasets
    Train_Data = DataGenerator(charge_obs[0:-splits], charge_hyp[0:-splits], batch_size=2 ** (15 + batch_scale),
                            )
    Val_Data = DataGenerator(charge_obs[-splits:-1], charge_hyp[-splits:-1], batch_size=2 ** (15 + batch_scale),
                             )

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        if args.use_relu:
            chargenet = get_chargenet(activation='relu')
        else:
            chargenet = get_chargenet()
        chargenet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], jit_compile=False)

    train_id = 'CHARGENET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

    #   Automatically train until validation loss does not decrease for 50 epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]
    #   additional callbacks for saving network as a function of epoch and extra analytics
    if args.save_history:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(save_freq='epoch', path_template=args.output_network[
                                                                                                 0] + 'resources/checkpoints/' + train_id + '{epoch:02d}'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.output_network[0] + 'resources/logs/' + train_id,
                                                        histogram_freq=1))

    hist = chargenet.fit(x=Train_Data,
                      validation_data=Val_Data,
                      epochs=int(args.epochs),
                      verbose=2,
                      callbacks=callbacks,
                      use_multiprocessing=True,
                      max_queue_size=512,
                      workers=2*n_gpus)

    # save the trained network
    tf.keras.models.save_model(chargenet, args.output_network[0] + '/chargenet', save_format='tf')
    tf.keras.models.save_model(chargenet, args.output_network[0] + '/chargenet.h5',
                               save_format='h5')  # Old format (HITMAN 0.1)

    # summarize history for loss and accuracy
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Chargenet Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.output_network[0] + '/chargenet.png', dpi=200)


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
    return parser.parse_args()


if __name__ == '__main__':
    main()
