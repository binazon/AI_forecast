from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import *
from datetime import *

'''
this method allows to  detect outliers in our dataset for each element of the data we substracted 
the mean and  we divide the result by the standard deviation

return the dataframe
'''
def detect_outliers(data):
    threshold, array, outlier =3, np.array(data), {}
    for j in range(len(array[0])):
        mean, std = np.mean(array[:, j]), np.std(array[:, j])
        tab= []
        for i in range(len(array[:, j])):
            z_score= (array[:, j][i] - mean)/std
            if np.abs(z_score) > threshold: tab.append(i)
        if(len(tab) != 0): outlier[j] = tab
    return data.iloc[sorted(set(sum([y for y in outlier.values()], [])))]

'''
this method allows to  detect outliers in our dataset with scipy.stats.z_score 
retunning dataframe
'''
def detect_outliers_2(df):
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[~filtered_entries]

'''
we build up the dataframe without outliers and we return a filtered dataframe
'''
def remove_outliers(df):
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[filtered_entries]