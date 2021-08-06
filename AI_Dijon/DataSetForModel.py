import os
import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.preprocessing import *
from Preprocessing import *

#creating the lstm dataset
def build_lstm_dataset(transformed_df, look):
    try:
        number_of_rows, number_of_features = transformed_df.shape[0], transformed_df.shape[1]
        train = np.empty([number_of_rows - look, look, number_of_features], dtype='float')
        label = np.empty([number_of_rows - look, 1])
        for i in range(number_of_rows-look):
            train[i] = transformed_df.iloc[i:i+look, 0:number_of_features]
            label[i] = transformed_df.iloc[i+look:i+look+1, 0:1]
    except Exception as error:
        print("Error in the dataset build :", error)
    return train, label 

#using data preprocessing - simple LSTM RNN Model
# -- each next value is based on look_back previous values
def prepare_data(timeseries_data, look_back):
    X, y = [],[]
    for i in range(len(timeseries_data)):
        end_ix = i + look_back
        if(end_ix >= len(timeseries_data)):
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x.flatten())
        y.append(seq_y[0])
    return np.array(X), np.array(y)