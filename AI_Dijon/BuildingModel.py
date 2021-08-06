import globals.Variable as global_vars
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import *

#######################################METHODS#####################################################################
#building the model RNN - LSTM
def buildModelLSTM():
    model = Sequential()
    model.add(LSTM(64, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'acc'])
    return model

#building the model - Recurrent Neural Network (LSTM) 
def buildModelLSTM1(neurons, dropout, weight_constraint, dijonX, dijonY):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(dijonX.shape[1], dijonX.shape[2]), kernel_constraint=MaxNorm(weight_constraint),
    return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(neurons//2, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(dijonY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model

'''
using the GRU model
'''
def buildModelGRU():
    model = Sequential()
    model.add(GRU(64, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model