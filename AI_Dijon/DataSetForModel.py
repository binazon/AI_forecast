import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import *

rootOutputFile = "files/output/"

'''
generating folders root path
'''
if not os.path.exists(rootOutputFile):os.makedirs(rootOutputFile)

#creating the lstm dataset
def create_lstm_dataset(df, look):
    try:
        number_of_rows, number_of_features = df.shape[0], df.shape[1]
        scaler = StandardScaler().fit(df.values)
        transformed_dataset = scaler.transform(df.values)
        transformed_df = pd.DataFrame(data = transformed_dataset, index=df.index)
        #writing the dijon trnsformed file
        f = open(rootOutputFile+"2- dijon_df_transformed.txt", "w+")
        f.write(str(transformed_df.head(50)))
        train = np.empty([number_of_rows - look, look, number_of_features], dtype='float')
        label = np.empty([number_of_rows - look, 1])
        for i in range(number_of_rows-look):
            train[i] = transformed_df.iloc[i:i+look, 0:number_of_features]
            label[i] = transformed_df.iloc[i+look:i+look+1, 0]
    finally:
        f.close()
    return scaler, train, label 

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