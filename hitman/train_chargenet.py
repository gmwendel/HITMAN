# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
def main():
    import argparse

    # Get command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.pkl', nargs='+',
                        required=True)
    parser.add_argument('-o', '--output_network',
                        help='Type = String; Output location for trained network, e.g. $PWD/chargenet.h5', nargs=1,
                        required=True)
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Type = Integer. Optional; limit number of epochs; Default = 1000')
    parser.add_argument('--use_relu', default=False, action="store_true",
                        help='Type = Boolean.  Optional; Use the relu activation function instead of mish; Default = False')
    parser.add_argument('--save_history', default=False, action="store_true",
                        help="Type = Boolean.  Optional; Add flag to save network at each epoch and enable Tensorboard stats in 'resource' folder ; Default = False")
    args = parser.parse_args()

    import datetime
    import numpy as np
    import tensorflow as tf

    from hitman.neural_nets.chargenet import get_chargenet
    from hitman.tools.datagenerator import DataGenerator

    import matplotlib.pyplot as plt
    import pickle


    # Load data for training

    prm = []
    hit = []
    chg = []
    cprm = []

    for file in args.input_files:
        fileObj = open(file, 'rb')
        events = pickle.load(fileObj)
        fileObj.close()
        for event in events:
            chg.append(event['total_charge'])
            cprm.append(event['truth'])
        del events
    print(len(chg))
    # build data in a form fit for training
    print('data converted')

    # Take 1/10 total data and make it validation
    splits = int(len(chg) / 10)

    Train_Data = DataGenerator(chg[0:-splits], cprm[0:-splits])
    Val_Data = DataGenerator(chg[-splits:-1], cprm[-splits:-1])

    labels = ['x', 'y', 'z', 'zenith', 'azimuth', 'time', 'energy']

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    if args.use_relu == False:
        chargenet = get_chargenet(labels)
    else:
        chargenet = get_chargenet(labels, activation='relu')
    chargenet.summary()

    optimizer = tf.keras.optimizers.Adam(0.005)
    chargenet.compile(loss='binary_crossentropy', optimizer=optimizer, jit_compile=True, metrics=['accuracy'])

    train_id = 'CHARGENET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50))
    if args.save_history == True:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(save_freq='epoch', path_template=args.output_network[
                                                                                                 0] + 'resources/checkpoints/' + train_id + '{epoch:02d}'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.output_network[0] + 'resources/logs/' + train_id,
                                                        histogram_freq=1))

    hist = chargenet.fit(x=Train_Data,
                         validation_data=Val_Data,
                         epochs=int(args.epochs),
                         verbose=2,
                         callbacks=callbacks,
                         use_multiprocessing=False,
                         max_queue_size=128)

    # summarize history for loss and accuracy

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.output_network[0] + '/chargenet.png', dpi=200)

    # save the trained network first in tf format and second in h5 format for backward compatibility
    tf.keras.models.save_model(chargenet, args.output_network[0] + '/chargenet', save_format='tf')
    tf.keras.models.save_model(chargenet, args.output_network[0] + '/chargenet.h5', save_format='h5')
