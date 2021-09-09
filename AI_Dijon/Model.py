import Globals.Variable as global_vars
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from xgboost import XGBRegressor

class Model:
    '''
    constructor of the class Model
    '''
    def __init__(self) -> None:
        pass

    #build vanilla LSTM model - ONE TO ONE model
    def vanillaLSTM(self):
        model = Sequential()
        model.add(LSTM(units=64, activation='relu', input_shape=(global_vars.x_shape, global_vars.y_shape)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae', 'acc'])
        return model, 'vanillaLSTM'

    #building the model RNN - LSTM
    def buildModelLSTM(self):
        model = Sequential()
        model.add(LSTM(160, input_shape=(global_vars.x_shape, global_vars.y_shape), kernel_constraint=MaxNorm(1),
        return_sequences=True))
        model.add(LSTM(3, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae', 'acc'])
        return model, 'deepLearningLSTM'

    #building default LSTM model
    def buildDefaultLSTM(self):
        model = Sequential()
        model.add(LSTM(72, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae', 'acc'])
        return model, 'defaultModelLSTM'

    '''
    using the Gated Recurent Unit RNN model
    '''
    def buildDefaultGRU(self):
        model = Sequential()
        model.add(GRU(64, input_shape=(global_vars.x_shape, global_vars.y_shape), return_sequences=True))
        model.add(GRU(32,return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae', 'acc'])
        return model, 'defaultModelGRU'

    '''
    using GBM : Gradient Boosting Machine model 
    '''
    def buildGBMModel(self):
        model = XGBRegressor(n_estimator = 500, learning_rate=0.001)
        return model, 'GBMModel'
