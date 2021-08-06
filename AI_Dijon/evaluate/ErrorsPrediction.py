import os.path
import numpy as np
import math
from sklearn.metrics import *
#######################################METHODS#####################################################################
#getting performances of the model
def generating_errors(test_values, predict, windows_size):
    #forecast errors
    forecast_error, eps, acc = np.array(test_values - predict), 1e-5, []
    print("forecast errors prediction : ", 
    ", ".join(np.array(abs(forecast_error[:,0][:windows_size])).astype('str')), "\n")
    print("Maximal error predict based on test : ", round(max(forecast_error[:,0]),2), "~", round(max(forecast_error[:,0])), "day(s)")
    test_values, predict = list(test_values.flatten()), list(predict.flatten())
    #precision of the model
    for i, y in enumerate(np.array(test_values)):
        acc.append(abs(np.array(predict)[i] - y) / y if abs(y) > eps else 0.0)
    acc_score = 100 - (100 * np.array(acc)).mean()
    print('precision of the regression model :', acc_score, "%")
    #bias or mean forecast error prediction
    print("BIAS or mean forecast error prediction :", (sum(forecast_error) * 1/len(test_values))[0])
    #mean abolute errors
    print("Mean absolute error (MAE) on test prediction :", mean_absolute_error(test_values, predict))
    #mean squared errors
    print("Mean squared error (MSE) on test prediction:", mean_squared_error(test_values, predict))
    #root mean squared errors
    print("Root mean squared error (RMSE) prediction or variance :", math.sqrt(np.mean(np.power(forecast_error[:,0],2))))
    #mean absolute percentage error
    '''tab=[]
    for i, y in enumerate(np.array(test_values)):
        tab.append(np.abs((y - np.array(predict)[i])/y) if abs(y) > eps else 0.0)
    print("Mean absolute percentage error (MAPE) :", np.mean(tab)*100)'''