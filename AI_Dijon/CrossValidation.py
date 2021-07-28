from matplotlib.pyplot import step
import numpy as np
from sklearn.model_selection import *
from sklearn.neighbors import *
from tensorflow.keras.wrappers.scikit_learn import *

'''
searching good hyperparameters for the model with gridSearchCV
'''
def fixHyperParamsGridSearch(model_model, x_train, y_train) -> str:
    model = KerasClassifier(build_fn=model_model, verbose=1)
    '''
    specifiying range of parameters
    '''
    neurons = [2,3,4,7]
    dropout_rate = [0,0.1,0.2, 0.3]
    weight_constraint = [0,1,2,3]
    optimizer = ['SGD', 'Adadelta', 'RMSprop', 'Adagrad', 'Adam']
    batch_size = np.arange(1,64)
    epochs = [500,1000,1500]
    '''
    model parameters and batch_size and epochs for the fit
    '''
    paramsGrid=dict(
        batch_size = list(batch_size),
        
    )
    grid = GridSearchCV(estimator=model, param_grid=paramsGrid, cv=3)
    grid_result = grid.fit(x_train, y_train)
    return 'best score : ' + str(grid_result.best_score_) + ' params : '  + str(grid_result.best_params_)