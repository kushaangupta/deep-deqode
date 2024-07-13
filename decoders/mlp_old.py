import pickle

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


class MLPOld:
    def __init__(self, in_dim=100, out_dim=2, hidden_dims=[], use_bias=True, args=None):
        self.model = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.getLayerOutput = [None] * 4

    def GetLayersOutput(self, x): 
        res = [x] 
        for i in range(4):
            res += [self.getLayerOutput[i]([x])[0]]
        
        return res

    def SaveWeights(self, path):
        d = {'w':self.model.get_weights(), 'in_dim': self.in_dim}
        with open(path, 'wb') as handle:
            pickle.dump(d, handle)

    def LoadWeights(self, path): 
        with open(path, 'rb') as handle:
            d = pickle.load(handle)

        w = d['w']
        self.in_dim = d['in_dim']

        self.CreateModel()
        self.model.set_weights(w)  

    def Predict(self, X):
        return self.model.predict(X)

    def Train(self, X_train, Y_train, X_val, Y_val):
        if(self.model is None):
            self.in_dim = X_train.shape[1]
            self.CreateModel()

        batch_size = 200
        nb_epoch   = 100

        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=10,
                verbose=0, restore_best_weights=True)
        ]

        self.model.fit(
            X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=0, validation_data=(X_val, Y_val), callbacks=callbacks)

    def CreateModel(self):
        model = Sequential() 

        model.add(Dense(100, input_shape=[self.in_dim], use_bias=self.use_bias))      
        model.add(Activation('relu'))

        model.add(Dense(50, use_bias=self.use_bias))
        model.add(Activation('relu'))

        model.add(Dense(25, use_bias=self.use_bias))      
        model.add(Activation('tanh'))        

        model.add(Dense(self.out_dim, use_bias=self.use_bias))
        model.add(Activation('linear'))        

        selectedLayers = [1, 3, 5, -1]

        for i in range(len(selectedLayers)):
            self.getLayerOutput[i] = K.function(
                [model.layers[0].input],  # , K.learning_phase()
                [model.layers[selectedLayers[i]].output]
            )

        ad = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=ad)    

        self.model  = model
