from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import *

#######################################METHODS#####################################################################
#building the model
def buildModel(neurons, dropout, weight_constraint, dijonX, dijonY):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(dijonX.shape[1],dijonX.shape[2]), kernel_constraint=MaxNorm(weight_constraint), return_sequences=True))
    model.add(LSTM(neurons//2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(dijonY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model

#building the model - Recurrent Neural Network (LSTM) 
def buildModel1(neurons, dropout, weight_constraint, dijonX, dijonY):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(dijonX.shape[1], dijonX.shape[2]), kernel_constraint=MaxNorm(weight_constraint),
    return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(neurons//2, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(dijonY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model