import math
from tensorflow.keras import *
from keras.layers import *
from keras.models import Sequential
#######################################METHODS#####################################################################

#building the model
def buildModel(enter, df, look):
    model = Sequential()
    model.add(LSTM(100, input_shape=(look,df.shape[1]), return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model

#building the model - Recurrent Neural Network (LSTM) 
def buildModel1(enter, df, look):
    model = Sequential()
    model.add(LSTM(55, input_shape=(look, df.shape[1]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(40, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model