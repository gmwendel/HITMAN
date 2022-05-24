#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
def main():
    
    import argparse

    #Get command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.pkl', nargs='+', required=True)
    parser.add_argument('-o', '--output_network', help='Type = String; Output location for trained network, e.g. $PWD/chargenet.h5', nargs=1, required=True)
    parser.add_argument('-g', '--output_graph', help='Type = String; Output location for loss(epoch) graph, e.g. $PWD/training_info.png', nargs=1, required=False)
    parser.add_argument('--epochs', default=1000, type=int, help='Type = Integer. Optional; limit number of epochs; Default = 1000')
    parser.add_argument('--t_shuffle', default=150, type=int, help='Type = Integer. Optional; Sets the standard deviation of the time shuffling; Default = 150')
    parser.add_argument('--use_mish', default=False, type=bool, help='Type = Boolean.  WARNING EXPERIMENTAL! Optional; Use the mish activation function; Default = False')
    args = parser.parse_args()
    
    import datetime
    import numpy as np
    import tensorflow as tf

    from hitman.neural_nets.chargenet import get_chargenet

    import matplotlib.pyplot as plt
    import pickle




    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, x, t, batch_size=2**12, shuffle='free', time_spread=50):
            assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."

            self.batch_size = int(batch_size/2) # half true labels half false labels
            self.data = np.array(x)
            self.params = np.array(t)

            #spread absolute time values (for hitnet)
            if len(self.data[0]) > 4:
                time_shifts = np.random.normal(0, time_spread, len(self.data))
                self.data[:, 3] += time_shifts
                self.params[:, 5] += time_shifts

            if shuffle == 'inDOM':
                self.shuffle_params_inDOM()
            else:
                self.shuffled_params = []

            self.indexes = np.arange(len(self.data))
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.data) / self.batch_size))

        def __getitem__(self, index):
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Generate data
            X, y = self.__data_generation(indexes)

            return X, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            np.random.shuffle(self.indexes) # mix between batches
            if len(self.shuffled_params) > 0:
                self.shuffle_params_inDOM()


        def __data_generation(self, indexes_temp):
            'Generates data containing batch_size samples'
            x = np.take(self.data, indexes_temp, axis=0)
            t = np.take(self.params, indexes_temp, axis=0)
            if len(self.shuffled_params) == 0:
                tr = np.random.permutation(t)
            else:
                tr = np.take(self.shuffled_params, indexes_temp, axis=0)

            d_true_labels = np.ones((self.batch_size, 1), dtype=x.dtype)
            d_false_labels = np.zeros((self.batch_size, 1), dtype=x.dtype)

            d_X = np.append(x, x, axis=0)
            d_T = np.append(t, tr, axis=0)
            d_labels = np.append(d_true_labels, d_false_labels)

            d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

            return (d_X, d_T), d_labels

        def unison_shuffled_copies(self, a, b, c):
            'Shuffles arrays in the same way'
            assert len(a) == len(b) == len(c)
            p = np.random.permutation(len(a))

            return a[p], b[p], c[p]


    #Load data for training
    events=[]
    for file in args.input_files:
        fileObj = open(file,'rb')
        events = pickle.load(fileObj)+events
        fileObj.close()
    print('data loaded')


    #build data in a form fit for training

    chg=[]
    cprm=[]
    print(len(events))
    for event in events:
        chg.append(event['total_charge'])
        cprm.append(event['truth'])
    print('data converted')

    del events


    #Take 1/10 total data and make it validation
    splits=int(len(chg)/10)

    Train_Data=DataGenerator(chg[0:-splits],cprm[0:-splits])
    Val_Data=DataGenerator(chg[-splits:-1],cprm[-splits:-1])

    labels=['x','y','z','zenith','azimuth','time','energy']



    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        if args.use_mish==True:
            chargenet = get_chargenet(labels)
        else:
            chargenet = get_chargenet(labels,activation='relu')
    chargenet.summary()


    optimizer = tf.keras.optimizers.Adam(0.0001)
    chargenet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_id = 'CHARGENET' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")


    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100))
#    callbacks.append(Save(save_every=2, path_template='resources/models/'+train_id+'/epoch_%i'))
#    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='resources/logs/'+train_id, histogram_freq=1))



    hist = chargenet.fit(x=Train_Data,
                            validation_data=Val_Data,
                            epochs=int(args.epochs),
                            verbose=2, 
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            max_queue_size=128)

    # summarize history for loss and accuracy
    if args.output_graph != None:
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(args.output_graph[0], dpi = 200)


    #save the trained network
    chargenet.save(args.output_network[0])
