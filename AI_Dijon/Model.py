import Globals.Variable as global_vars
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import *
from tensorflow.keras.optimizers import *


class Model:
    '''
    constructor of the class Model
    '''
    def __init__(self) -> None:
        pass

    #building the model RNN - LSTM
    def buildModelLSTM(self, neurons):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
        model.add(LSTM(32,return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae', 'acc'])
        return model

    '''
    using the Gated Recurent Unit RNN model
    '''
    def buildModelGRU(self):
        model = Sequential()
        model.add(GRU(64, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
        model.add(GRU(32,return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])
        return model
