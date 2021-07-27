import os
from scipy.ndimage.measurements import label
from scipy.stats import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import *
import time
from scipy.stats.stats import zscore
from sklearn import linear_model
plt.style.use('seaborn')

pathInput, pathInputAnalysis, pathOutput = "generated/graphs/input/samples/", "generated/graphs/input_analysis/outliers/", "generated/graphs/output/"
pathInputRegression = "generated/graphs/input/linear_regression/"
'''
generating folders root path
'''
if not os.path.exists(pathInput):os.makedirs(pathInput)
if not os.path.exists(pathInputRegression):os.makedirs(pathInputRegression)
if not os.path.exists(pathInputAnalysis):os.makedirs(pathInputAnalysis)
if not os.path.exists(pathOutput):os.makedirs(pathOutput)

'''
saving in graphs/ folder the graphs nbDi or meteo by date
'''
def graphNbDiMeteoByDate(df):
    nb_days = 90
    try:
        for i in range(1, len(df.columns)):
            if(i==1):
                '''
                nbDi by date the 3 last month -- get more details
                '''
                try:
                    fig, ax = plt.subplots(figsize=(24,11))
                    plt.title(str(nb_days)+" last_days_nbDi_by_date")
                    plt.axhline(y=10, linewidth=2, color='black', label='limit weekend nbDi', linestyle='--')
                    ax.text(0, 38, 'weekends in red on x axis, nbDi(weekend) < 10 (seasonality)', style='normal',fontsize=20, bbox={'facecolor': 'none',
                    'alpha': 0.5, 'pad': 10})
                    ax.plot(*zip(*sorted(zip(df[["date"]].values.flatten()[-nb_days:],df[df.columns[1]][-nb_days:].astype(float)))),'-o', color='blue', label=df.columns[1])
                    ax.set_xticks(range(len(df[["date"]].values.flatten()[-nb_days:])))
                    ax.set_xticklabels(df[["date"]].values.flatten()[-nb_days:], rotation=90)
                    for xtick in ax.get_xticklabels():
                        if(pd.to_datetime(xtick.get_text()).weekday()>4):xtick.set_color("red")
                    plt.xlabel("date",fontsize=14)
                    plt.ylabel(df.columns[1],fontsize=14)
                    plt.legend()
                    plt.savefig(pathInput+'1.'+str(1)+ '- '+df.columns[1]+'_by_date_last_3_month.png')
                finally:
                    plt.close()                
            #plotting all feature of the dataframe by date
            plt.subplots(figsize=(24,11))
            plt.title(df.columns[i]+' by date')
            plt.plot(*zip(*sorted(zip(df[["date"]].values.flatten(),df[df.columns[i]].astype(float)))), color='blue', label=df.columns[i])
            plt.xticks(df.index, df[["date"]].values.flatten(), rotation=90)
            plt.locator_params(axis='x', nbins=15)
            plt.xlabel("date",fontsize=14)
            plt.ylabel(df.columns[i],fontsize=14)
            plt.legend()
            plt.savefig(pathInput+'1.'+str(i)+ '- '+df.columns[i]+'_by_date.png')
    finally:
        plt.close()

'''
plotting the graph of linear regression of nbDi by date
'''
def linearRegressionNbDiByDate(df):
    nbDi_column = df.columns[1]
    regress = linear_model.LinearRegression()
    '''
    date to timetamp
    '''
    timestamp = pd.DataFrame([time.mktime(datetime.strptime(i, "%Y-%m-%d").timetuple()) for i in np.array(df['date'])])    
    regress.fit(timestamp, df[nbDi_column])
    res = regress.predict(timestamp)
    try:
        #plotting all feature of the dataframe by date
        plt.subplots(figsize=(24,11))
        plt.title('linear regression of '+nbDi_column+' by date')
        plt.scatter(*zip(*sorted(zip(df[["date"]].values.flatten(),df[nbDi_column].astype(float)))), color='blue', marker='+', label=nbDi_column)
        #plt.plot(*zip(*sorted(zip(timestamp, res ))), color='black')
        plt.xticks(df.index, df[["date"]].values.flatten(), rotation=90)
        plt.locator_params(axis='x', nbins=15)
        plt.xlabel("date",fontsize=14)
        plt.ylabel(nbDi_column,fontsize=14)
        plt.legend()
        plt.savefig(pathInputRegression+'linear_regression_'+nbDi_column+'_by_date.png')
    finally:
        plt.close()



'''
getting the history model : train_losses, train_accuracy, val_losses, val_accuracy
'''
def graphHistoryModel(history):
    try:
        plt.subplots(figsize=(18,9))
        plt.plot(history.history['loss'], label = 'train_losses')
        plt.plot(history.history['acc'], label = 'train_accuracy')
        plt.plot(history.history['val_loss'], label='val_losses')
        plt.plot(history.history['val_acc'], label='val_accuracy')
        plt.xlabel("epoch",fontsize=14)
        plt.ylabel("validation and train values",fontsize=14)
        plt.legend()
        plt.savefig(pathOutput+'2- history_train.png')
    finally:
        plt.close()

'''
plotting truth and nbDi prediction

sign of nb_elmnts_to_print design the kind of prediction on thruth will be plotted
positive nb_elmnts_to_print : plot nb_elmnts_to_print first elements
negative nb_elmnts_to_print : plot nb_elmnts_to_print last elements
'''
def graphTruthOnPrediction(nb_elmnts_to_print, y_test, test_predict):
    try:
        plt.subplots(figsize=(18,9))
        plt.ylabel("nb demandes d'intervention",fontsize=14)
        plt.xlabel("pas par date",fontsize=14)
        if(nb_elmnts_to_print >0) :
            '''
            plotting nb_elmts_to_print first : predicted on truth values
            ''' 
            plt.title('prediction sur valeurs réelles : ('+str(nb_elmnts_to_print)+' premiers éléments) on '+str(len(y_test))+' elements total')
            plt.plot(y_test[:nb_elmnts_to_print],'-o', color="blue", label="réel nbDi")
            plt.plot(test_predict[:nb_elmnts_to_print],'-o', color="green",label="prédiction nbDi")
            plt.legend()
            plt.savefig(pathOutput+'3- predictOnTest_'+str(nb_elmnts_to_print)+'_first.png')
        elif(nb_elmnts_to_print < 0):
            '''
            plotting nb_elmts_to_print last : predicted on truth values
            '''
            plt.title('prediction sur valeurs réelles : ('+str(abs(nb_elmnts_to_print))+' derniers éléments) sur '+str(len(y_test))+' elements total')
            plt.plot(y_test[nb_elmnts_to_print:],'-o', color="blue", label="réel nbDi")
            plt.plot(test_predict[nb_elmnts_to_print:],'-o', color="green",label="prédiction nbDi")
            plt.legend()
            plt.savefig(pathOutput+'3- predictOnTest_'+str(abs(nb_elmnts_to_print))+'_last.png')
    finally:
        plt.close()

'''
predict feature : (nb_days_predict) days
'''
def graphPredictNextDays(last_data, NB_DAYS_PREDICTED, dijon, feature, dijon_timestamps, dijon_dates):
    try:
        plt.subplots(figsize=(24,11))
        plt.xticks(rotation=90)
        plt.xlabel("date de demande d'intervention",fontsize=14)
        plt.ylabel("nombre de demande d'intervention",fontsize=14)
        plt.title(str(last_data)+' derniers jours + ' + 'prédiction nbDi par date ('+ str(NB_DAYS_PREDICTED)+' futur jours)')
        plt.axvline(x=last_data -1,ymin=0,ymax=max(dijon['nbDi']), linewidth=3, color='black', label='current day', linestyle='--')
        #getting 2 last month values for
        dijon_labels = np.array(pd.DataFrame(dijon['nbDi'], dtype='float').tail(last_data)).flatten()
        #plotting features values
        plt.plot(*zip(*sorted(zip(dijon_timestamps,dijon_labels))), color='blue', label='truth ' + str('in 62 last days'))
        plt.plot(*zip(*sorted(zip(dijon_dates, feature))), color='orange', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
        plt.legend()
        plt.savefig(pathOutput+'4- last '+str(last_data)+' jours + predict_'+str(NB_DAYS_PREDICTED)+'_next_days.png')
    finally:
        plt.close()

'''
graph with just feature informations
'''
def graphFeatureInfos(dijon_dates, feature, NB_DAYS_PREDICTED):
    try:
        plt.subplots(figsize=(24,11))
        plt.xticks(rotation=90)
        plt.bar(dijon_dates, feature, color='orange', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
        plt.title(str(NB_DAYS_PREDICTED) + ' next predictions values (bar chart)')
        plt.xlabel("date de demande d'intervention",fontsize=14)
        plt.ylabel("nombre de demande d'intervention",fontsize=14)
        plt.legend()
        plt.savefig(pathOutput+'5- predict_'+str(NB_DAYS_PREDICTED)+'_next_days.png')
    finally:
        plt.close()

'''
this graph allows to show the z_score that we get for eatch value of nbDi dataset

values thaht are higher than 3 are the outliers.
'''
def graphZScoreByDate(df):
    df_without_date = df.iloc[:, 1:].astype('float')
    z_scores = pd.DataFrame(zscore(df_without_date))
    try:
        for i in range(df_without_date.shape[1]):
            #plotting all feature of the dataframe by date
            plt.subplots(figsize=(24,11))
            plt.axhline(y=3, linewidth=2, color='r', label='limit z_score == 3')
            plt.text(0, 10, 'value is outlier if z_score(value) > limit z_score', style='normal',fontsize=30, bbox={'facecolor': 'none','alpha': 0.5, 'pad': 10})
            plt.title('z_score '+df_without_date.columns[i]+' by date')
            plt.plot(*zip(*sorted(zip(df[["date"]].values.flatten(),df_without_date[df_without_date.columns[i]].astype(float)))), color='blue', label=df_without_date.columns[i])
            plt.plot(*zip(*sorted(zip(df[["date"]].values.flatten(),z_scores.iloc[:,i]))), color='black', label="z_score "+df_without_date.columns[i])
            plt.xticks(df.index, df[["date"]].values.flatten(), rotation=90)
            plt.locator_params(axis='x', nbins=15)
            plt.xlabel("date",fontsize=14)
            plt.ylabel("z_score_"+df_without_date.columns[i],fontsize=14)
            plt.legend()
            plt.savefig(pathInputAnalysis+'1.'+str(i)+ '- z_score_'+df_without_date.columns[i]+'_by_date.png')
    finally:
        plt.close()

