import numpy as np
from sklearn.model_selection import *
from sklearn.neighbors import *
from tensorflow.keras.wrappers.scikit_learn import *
import matplotlib.pyplot as plt

'''
searching good hyperparameters for the model with gridSearchCV
'''
def fixHyperParamsGridSearch(model_model, x_train, y_train) -> str:
    model = KerasClassifier(build_fn=model_model, verbose=1)
    '''
    specifiying range of parameters
    '''
    neurons = np.arange(x_train.shape[1], x_train.shape[1] * 3, step=x_train.shape[1])
    dropout_rate = np.arange(0, 0.2, step=0.1)
    weight_constraint = np.arange(0, 3)
    batch_size = np.arange(32,64, step=32)
    epochs = np.arange(500, 1500, step=500)
    '''
    model parameters and batch_size and epochs for the fit
    '''
    paramsGrid=dict(
        neurons = list(neurons),
        dropout = list(dropout_rate),
        weight_constraint = list(weight_constraint), 
        dijonX = list(x_train),
        dijonY = list(y_train),
        batch_size = list(batch_size),
        epochs = list(epochs)
    )
    grid = GridSearchCV(estimator=model, param_grid=paramsGrid)

    print(x_train.shape)
    print(y_train.shape)
    try:
        grid.fit(x_train, y_train)
    except ValueError:
        pass
    #return 'best score : ' + str(grid_result.best_score_) + 'params : '  + str(grid_result.best_params_)