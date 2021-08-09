from typing import List
import numpy as np
import pandas as pd
from Preprocessing import *

'''
class for prediction methods and attributes
'''
class Prediction :

    '''
    constructor
    '''
    def __init__(self, model, array_enter, nbdays, look_back,):
        self.model = model
        self.array_enter = array_enter
        self.nbdays = nbdays
        self.look_back = look_back
        self.processing = Preprocessing()

    #predict data in the next nbdays
    def predictNextDays(self, futureMeteo) -> List:
        x_input = np.array(pd.DataFrame(self.array_enter).tail(self.look_back), dtype='float')
        x_input = self.processing.normaliseData(x_input)
        temp_input, lst_output, i = list(x_input), [], 1
        while(i<=self.nbdays):
            if(len(temp_input) > self.look_back):
                x_input = np.array(temp_input[1:], dtype='float')
                x_input = x_input.reshape(1,self.look_back, x_input.shape[1])
                yhat = self.model.predict(x_input)
                for p in futureMeteo.head(i).to_numpy().flatten()[1:futureMeteo.shape[1]]:yhat = np.append(yhat, p)
                yhat = yhat.reshape(1,self.array_enter.shape[1])
                yhat = self.processing.unormaliseData(yhat)[0]
                temp_input.append(yhat)
                temp_input = temp_input[1:]
            else:
                x_input = x_input.reshape(1,self.look_back, x_input.shape[1])
                yhat = self.model.predict(x_input)
                for p in futureMeteo.head(i).to_numpy().flatten()[1:futureMeteo.shape[1]]:yhat = np.append(yhat, p)
                yhat = yhat.reshape(1,self.array_enter.shape[1])
                yhat = self.processing.unormaliseData(yhat)[0]
                temp_input.append(yhat)
            lst_output.append(yhat[0])
            i=i+1
        secondElmtToLast=np.array(lst_output).reshape(-1,1)
        return secondElmtToLast.reshape(secondElmtToLast.shape[0])