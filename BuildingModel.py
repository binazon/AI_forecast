import math
from tensorflow.keras import *
from keras.layers import *
from keras.models import Sequential
#######################################METHODS#####################################################################

#building the model
def buildModel(enter, data_train, data_label):
    model = Sequential()
    model.add(LSTM(150, input_shape=(data_train.shape[1],data_train.shape[2]), return_sequences=True))
    #model.add(Dropout(0.3))
    model.add(LSTM(70, return_sequences=False))
    #model.add(Dropout(0.3))
    model.add(Dense(data_label.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model


#building the model - Recurrent Neural Network (LSTM) 
def buildModel1(enter, data_train, data_label):
    model = Sequential()
    model.add(LSTM(55, input_shape=(data_train.shape[1],data_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(40, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(data_label.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mae'])
    return model