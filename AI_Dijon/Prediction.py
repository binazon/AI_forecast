from typing import List
import numpy as np
import pandas as pd
#######################################METHODS#####################################################################
#predict data in the next nbdays
def predictNextDays(model, array_enter, nbdays, scaled, look_back, futureMeteo) -> List:
    x_input = np.array(pd.DataFrame(array_enter).tail(look_back), dtype='float')
    temp_input, lst_output = list(x_input), [x_input[len(x_input)-1]]
    i = 1
    while(i<=nbdays):
        if(len(temp_input) > look_back):
            x_input = np.array(temp_input[1:], dtype='float')
            x_input = x_input.reshape(1,look_back, x_input.shape[1])
            yhat = model.predict(x_input, batch_size=32, verbose = 2)
            for p in futureMeteo.head(i).to_numpy().flatten()[1:futureMeteo.shape[1]]:yhat = np.append(yhat, p)
            yhat = yhat.reshape(1,array_enter.shape[1])
            yhat = scaled.inverse_transform(yhat)[:,0]
            temp_input.append(yhat[0])
            temp_input = temp_input[1:]
        else:
            x_input = x_input.reshape(1,look_back, x_input.shape[1])
            yhat = model.predict(x_input, batch_size=32, verbose = 2)
            for p in futureMeteo.head(i).to_numpy().flatten()[1:futureMeteo.shape[1]]:yhat = np.append(yhat, p)
            yhat = yhat.reshape(1,array_enter.shape[1])
            yhat = scaled.inverse_transform(yhat)[:,0]
            temp_input.append(yhat[0])
        lst_output.append(max(0,yhat[0][0]))
        i=i+1
    secondElmtToLast=np.array(lst_output[1:]).reshape(-1,1)
    return secondElmtToLast.reshape(secondElmtToLast.shape[0])