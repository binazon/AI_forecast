import os.path
import numpy as np
import math
#######################################METHODS#####################################################################
#getting performances of the model
def generating_errors(test, predict, windows_size):
    #forecast errors
    forecast_error = np.array(test - predict)
    forecast_errors = np.array([abs(test[i]-predict[i]) for i in range(len(test))])
    print("forecast errors prediction ", windows_size,
    "first elements : ".join(np.array(forecast_errors[:,0][:windows_size]).astype('str')))
    print("maximal forecast error : "+ str(max(forecast_errors[:,0]))+"\n")
    #mean forecast error
    print("mean forecast error prediction : ", np.mean(forecast_error))
    #mean abolute errors
    print("mean absolute error prediction : ", np.mean(abs(forecast_error)))
    #mean squared errors
    mse_prediction = np.mean(np.power(forecast_error[:,0],2))
    print("mean squared error prediction : ", mse_prediction)
    #root mean squared errors
    print("root mean squared error prediction : ", math.sqrt(mse_prediction))