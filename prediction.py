from typing import List
import numpy as np
import pandas as pd

#predict data in the next nbdays
def predictNextDays(model_entry, array_enter, nbdays, scaled, look_back) -> List:
    x_input = np.array(pd.DataFrame(array_enter).tail(look_back), dtype='float')
    temp_input = list(x_input)
    lst_output = [x_input[len(x_input)-1]]
    i = 0
    while(i<nbdays):
        if(len(temp_input) > look_back):
            x_input = np.array(temp_input[1:], dtype='float')
            x_input = x_input.reshape(1,look_back, x_input.shape[1])
            yhat = model_entry.predict(x_input, batch_size=32, verbose = 2)
            yhat = scaled.inverse_transform(yhat)
            #print("{} day output {}".format(i, yhat[0][0]))
            temp_input.append(yhat[0])
            temp_input = temp_input[1:]
        else:
            x_input = x_input.reshape(1,look_back, x_input.shape[1])
            yhat = model_entry.predict(x_input, batch_size=32, verbose = 2)
            yhat = scaled.inverse_transform(yhat)
            #print("{} day output {} ".format(i,yhat[0][0]))
            temp_input.append(yhat[0])
        lst_output.append(max(0,yhat[0][0]))
        i=i+1
    secondElmtToLast=np.array(lst_output[1:]).reshape(-1,1)
    return secondElmtToLast.reshape(secondElmtToLast.shape[0])