from matplotlib.pyplot import step
import numpy as np
from sklearn.model_selection import *
from tensorflow.keras.wrappers.scikit_learn import *

'''
searching good hyperparameters for the model with gridSearchCV
'''
def fixHyperParamsGridSearch(model_model, x_train, y_train) -> str:
    model = KerasRegressor(build_fn=model_model)
    '''
    specifiying range of parameters
    '''
    dropout_rate = [0.0,0.1,0.2, 0.3]
    epochs = [500,1000]
    activation = ['relu', 'softmax', 'sigmoid',  'tanh']
    batch_size = np.arange(1,64, step=10)
    optimizer = ['Adam','Adadelta', 'RMSprop', 'SGD', 'Adagrad']
    weight_constraint = [0,1,2,3]
    neurons = np.arange(156, 161)
    learning_rate = [1e-2, 1e-3, 1e-4]
    kernel_regularizer = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    '''
    model parameters and batch_size and epochs for the fit
    '''
    paramsGrid=dict(
        neurons = list(neurons)
    )
    grid = GridSearchCV(estimator=model, param_grid=paramsGrid, cv=4)
    grid.fit(x_train, y_train)
    return 'best score : ' + str(grid.best_score_) + ' params : '  + str(grid.best_params_)
