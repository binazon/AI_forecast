import os.path
import numpy as np
import math
from sklearn.metrics import *
#######################################METHODS#####################################################################
#getting performances of the model
def generating_errors(test_values, predict, windows_size):
    #forecast errors
    forecast_error = np.array(test_values - predict)
    print("forecast errors prediction : ", 
    ", ".join(np.array(forecast_error[:,0][:windows_size]).astype('str')), "\n")
    print("Maximal error predict based on test : ", round(max(forecast_error[:,0]),2), "~", round(max(forecast_error[:,0])), "day(s)")
    test_values = list(test_values.flatten())
    predict = list(predict.flatten())
    print('Total precision : ', round(precision_score([round(i) for i in test_values], [round(i) for i in predict], average='micro'),2), "%")
    #bias or mean forecast error prediction
    print("BIAS or mean forecast error prediction : ", (sum(forecast_error) * 1/len(test_values))[0])
    #mean abolute errors
    print("Mean absolute error (MAE) on test prediction : ", mean_absolute_error([round(i) for i in test_values], [round(i) for i in predict]))
    #mean squared errors
    print("Mean squared error (MSE) on test prediction: ", mean_squared_error([round(i) for i in test_values], [round(i) for i in predict]))
    #root mean squared errors
    mse_prediction = np.mean(np.power(forecast_error[:,0],2))
    print("Root mean squared error (RMSE) prediction : ", math.sqrt(mse_prediction))
