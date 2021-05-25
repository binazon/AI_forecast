import math
from tensorflow.keras import *
from keras.layers import *
from keras.models import Sequential
#######################################METHODS#####################################################################
#triying to build the model
def buildModel(enter, data):
    model = Sequential()
    model.add(LSTM(55, dropout= 0.02, batch_input_shape=(None,data.shape[1],data.shape[2]), return_sequences=True))
    model.add(LSTM(40, dropout=0.02, activity_regularizer=regularizers.l2(1e-5), return_sequences=False))
    model.add(Dense(data.shape[2]))
    model.compile(optimizer='adam', loss='mse', metrics=['acc', 'mae'])
    return model