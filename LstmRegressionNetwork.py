# import io
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.optimizers import Adam
# from keras.layers import Flatten
from keras.layers import LSTM, GRU, GRUCell
# import matplotlib.pyplot as plt
# import seaborn as sns
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
import datetime

class LstmRegressionNetwork:
    # RNN class to avoid annoying coding while exploring hyper parameters
    def __init__(self, rnn_units=[1, 1], dense_units=[4, 4], n_out=1, 
                 lr=0.001, decay=0.0025, 
                 l2_rnn_kernel=0.001, l2_rnn_activity=0.001, l2_fc=0.001, 
                 clipnorm=1.0, epochs=3, batch_size=2**10, 
                 stateful=False, shuffle=True, 
                 verbose=1):
      
        # hyper parameters
        self.lr = lr
        self.decay = decay
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.n_out = n_out
        self.l2_rnn_kernel = l2_rnn_kernel
        self.l2_rnn_activity = l2_rnn_activity
        self.l2_fc = l2_fc
        self.clipnorm = clipnorm
        self.epochs = epochs
        self.batch_size = batch_size
        self.stateful = stateful
        self.shuffle = shuffle

        # placeholder for model
        self.model = None
        self.verbose = verbose

        # place to store model history output
        self.History = None
        self.loss = None
        self.val_loss = None


    def build_model(self, x_train):
        """
        Build the LSTM regression model

        :param x_train: sequance data set of shape (n, T, F)
        :return: none, sets self.model
        """
        # setup params
        n_steps = x_train.shape[1]

        n_feats = x_train.shape[2]

        # define model
        model = Sequential()

        # setup regularizers
        l2_rnn_kernel = regularizers.l2(self.l2_rnn_kernel)
        l2_rnn_activity = regularizers.l2(self.l2_rnn_activity)
        l2_fc = regularizers.l2(self.l2_fc)

        # return the activation values at each timestep as a sequence
        for i in range(len(self.rnn_units)):
            model.add(LSTM(self.rnn_units[i],
                           activation='tanh',
                           input_shape=(None, n_feats),
                           batch_input_shape=(self.batch_size, n_steps, n_feats),
                           return_sequences=True,
                           stateful=self.stateful,
                           kernel_regularizer=l2_rnn_kernel,
                           activity_regularizer=l2_rnn_activity))
            
        if self.dense_units is not None:
          # apply a single set of dense layers to each timestep to produce a sequence
          for i in range(len(self.dense_units)):
              if i == 0:
                  model.add(TimeDistributed(Dense(self.dense_units[i], 
                                                  activation='relu',
                                                 kernel_regularizer=l2_fc)))
              else:
                  model.add(Dense(self.dense_units[i], activation='relu', 
                                  kernel_regularizer=l2_fc))
                  
                  # add output layer
                  model.add(Dense(self.n_out))
        else:
          model.add(TimeDistributed(Dense(self.n_out)))

        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=self.decay, amsgrad=False,
                              clipnorm=self.clipnorm)

        model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

        self.model = model

    def fit_model(self, x_train, y_train, x_val, y_val):
        """
        train the model

        :param x_train: numpy array (n, t, f)
        :param y_train: numpy array (n, t)
        :param x_val: numpy array (n, t, f)
        :param y_val: numpy array (n, t)
        :return: none, fits self.model
        """
        start_time = datetime.datetime.now()
        
        if self.stateful:
          self.loss = []
          self.val_loss = []
          
          for i in range(self.epochs):
            self.History = self.model.fit(x=x_train, y=y_train,
                                        validation_data=[x_val, y_val],
                                        epochs=1, 
                                        batch_size=self.batch_size,
                                        verbose=self.verbose,
                                        shuffle=self.shuffle)
            
            self.loss.append(self.History.history['loss'])
            self.val_loss.append(self.History.history['val_loss'])
            
            self.model.reset_states()
           
          self.loss = np.array(self.loss).ravel()
          self.val_loss = np.array(self.val_loss).ravel()
        else:
          self.History = self.model.fit(x=x_train, y=y_train,
                                        validation_data=[x_val, y_val],
                                        epochs=self.epochs, 
                                        batch_size=self.batch_size,
                                        verbose=self.verbose,
                                        shuffle=self.shuffle)
          
          self.loss = self.History.history['loss']
          self.val_loss = self.History.history['val_loss']

        print('train time: ' + 
              str((datetime.datetime.now() - start_time)))

    def reset_weights(self):
        """
        reset the weights to try training new models from scratch

        :return: none, re-initiallizes weights in self.model
        """
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
