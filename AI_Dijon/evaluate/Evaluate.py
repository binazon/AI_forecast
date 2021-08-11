from scipy.stats.stats import mode
import os.path
import numpy as np
import math
from sklearn.metrics import *

'''
this class allow to evaluate the model used and predictions
'''
class Evaluate :

    '''
    definition of the constructor
    '''
    def __init__(self, model):
        self.model = model

    '''
    evaluate de model performances
    '''
    def evaluateModel(self, dijon_train, label_train, dijon_test, label_test) -> str:
        #evaluation in train dataset
        eval_train = self.model.evaluate(dijon_train, label_train)
        model_eval ="Pertes sur le train : "+str(eval_train[0]) + "\n"
        model_eval+="erreure absolue moyenne (MAE) sur train : "+str(eval_train[1]) + "\n"
        #evaluation in test dataset
        eval_test = self.model.evaluate(dijon_test, label_test)
        model_eval+="Pertes sur le test : "+str(eval_test[0]) + "\n"
        model_eval+="erreure absolue moyenne (MAE) sur test : "+str(eval_test[1]) + "\n"
        model_eval+="\n"+"Votre model généralise." if abs(eval_train[1]*100 - eval_test[1]*100) <= 5 else "Votre model a tendance à suraprendre.\n"
        return model_eval

    '''
    getting the model on prediction
    '''
    def generating_errors(self, truth, predict, windows_size):
        #forecast errors
        forecast_error, eps, acc = np.array(truth - predict), 1e-5, []
        print("Maximal error predict based on truth : ", round(max(forecast_error[:,0]),2), "~", round(max(forecast_error[:,0])), "day(s)")
        truth, predict = list(truth.flatten()), list(predict.flatten())
        #precision of the model
        for i, y in enumerate(np.array(truth)):
            acc.append(abs(np.array(predict)[i] - y) / y if abs(y) > eps else 0.0)
        acc_score = 100 - (100 * np.array(acc)).mean()
        print('precision of the regression model :', acc_score, "%")
        #bias or mean forecast error prediction
        print("BIAS or mean forecast error prediction :", (sum(forecast_error) * 1/len(truth))[0])
        #mean abolute errors
        print("Mean absolute error (MAE) on truth prediction :", mean_absolute_error(truth, predict))
        #mean squared errors
        print("Mean squared error (MSE) on truth prediction:", mean_squared_error(truth, predict))
        #root mean squared errors
        print("Root mean squared error (RMSE) prediction or variance :", math.sqrt(np.mean(np.power(forecast_error[:,0],2))))
        #mean absolute percentage error
        '''tab=[]
        for i, y in enumerate(np.array(truth)):
            tab.append(np.abs((y - np.array(predict)[i])/y) if abs(y) > eps else 0.0)
        print("Mean absolute percentage error (MAPE) :", np.mean(tab)*100)'''