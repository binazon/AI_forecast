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
    print('Total precision : ', precision_score([round(i) for i in test_values], [round(i) for i in predict], average='micro'), "%")
    #bias or mean forecast error prediction
    print("BIAS or mean forecast error prediction : ", (sum(forecast_error) * 1/len(test_values))[0])
    #mean abolute errors
    print("Mean absolute error (MAE) on test prediction : ", mean_absolute_error(test_values, predict))
    #mean squared errors
    print("Mean squared error (MSE) on test prediction: ", mean_squared_error(test_values, predict))


    #calculate accuracy
    '''inv_yhat = np.array(predict)
    inv_y = np.array(test_values)

    res, tab = 0, []
    for i in inv_y:
        if(i == 0):
            res = 0
        else:
            res = (100 * (abs(inv_yhat - inv_y) / inv_y))
        tab.append(res)

    acc =100 - np.array(tab).mean()
'''



    eps, acc = 1e-5, []
    for i, y in enumerate(np.array(test_values)):
        if abs(y) > eps:
            acc.append(abs(np.array(predict)[i] - y) / y)
        else:
            acc.append(0.0)
    acc_score = 100 - (100 * np.array(acc)).mean()
        
    #acc = 100 - (100 * (abs(inv_yhat - inv_y) / inv_y)).mean()

    print('accuracy', acc_score)




    #root mean squared errors
    mse_prediction = np.mean(np.power(forecast_error[:,0],2))
    print("Root mean squared error (RMSE) prediction : ", math.sqrt(mse_prediction))
