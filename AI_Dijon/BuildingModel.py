import math
from tensorflow.keras import *
from keras.layers import *
from keras.models import Sequential
#######################################METHODS#####################################################################

#building the model
def buildModel(enter, dijonX, dijonY):
    model = Sequential()
    model.add(LSTM(dijonX.shape[1], input_shape=(dijonX.shape[1],dijonX.shape[2]), return_sequences=True))
    model.add(LSTM(dijonX.shape[1], return_sequences=False))
    model.add(Dense(dijonY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model

#building the model - Recurrent Neural Network (LSTM) 
def buildModel1(enter, dijonX, dijonY):
    model = Sequential()
    model.add(LSTM(55, input_shape=(dijonX.shape[1], dijonX.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(40, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(dijonY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model