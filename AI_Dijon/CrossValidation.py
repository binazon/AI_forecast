import numpy as np
from sklearn.model_selection import *
from sklearn.neighbors import *
import matplotlib.pyplot as plt

'''
search model's input hyperparameter with validation_curve
'''
def getNeighborsGraphStats(x_train, y_train):
    model = KNeighborsClassifier()
    k = np.arange(1, 15,9)
    train_score, val_score = validation_curve(model,x_train, y_train, param_name='n_neighbors', param_range=k, cv=5)
    #plt.plot(k, val_score.mean(axis=1), label='validation')
    #plt.plot(k, train_score.mean(axis=1), label='validation')
    
'''
analyse model with grid search cv
'''
def getGrid(x_train, y_train) -> str:
    param_grid={
        'n_neighbors':np.arange(1, 50),
        'metric' : ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(x_train, y_train)
    return 'best score : ' + str(grid.best_score_) + 'params : '  + str(grid.best_params_)
