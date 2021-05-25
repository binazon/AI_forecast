import os.path
import numpy as np
import math
#######################################METHODS#####################################################################
#performances of the model
def generating_errors(test, predict, windows_size):
    #forecast errors
    forecast_error = np.array(test - predict)
    forecast_errors = np.array([test[i]-predict[i] for i in range(len(test))])
    print("forecast errors prediction: ", forecast_errors[:,0][:windows_size])
    #mean forecast error
    print("mean forecast error prediction: ", np.mean(forecast_error), "")
    #mean abolute errors
    print("mean absolute error prediction: ", np.mean(abs(forecast_error)), "")
    #mean squared errors
    mse_prediction = np.mean(np.power(forecast_error[:,0],2))
    print("mean squared error prediction: ", mse_prediction, "")
    #root mean squared errors
    print("root mean squared error prediction: ", math.sqrt(mse_prediction), "")