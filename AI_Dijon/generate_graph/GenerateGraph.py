import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pathInput, pathOutput = "imgs/input/", "imgs/output/"
'''
generating folders root path
'''
if not os.path.exists(pathInput):os.makedirs(pathInput)
if not os.path.exists(pathOutput):os.makedirs(pathOutput)

'''
saving in imgs folder the graphs nbDi or meteo by date
'''
def graphNbDiMeteoByDate(df):
    try:
        for i in range(1, len(df.columns)):
            plt.subplots(figsize=(24,11))
            plt.plot(*zip(*sorted(zip(df[["date"]].values.flatten(),df[df.columns[i]].astype(float)))), color='blue', label=df.columns[i])
            plt.xticks(df.index, df[["date"]].values.flatten(), rotation=90)
            plt.locator_params(axis='x', nbins=15)
            plt.xlabel("date",fontsize=14)
            plt.ylabel(df.columns[i],fontsize=14)
            plt.legend()
            plt.savefig(pathInput+'1.'+str(i)+ '- '+df.columns[i]+'_byDate.png')
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
'''
def graphTruthOnPrediction(nb_elmnts_to_print, y_test, test_predict):
    try:
        fig,ax=plt.subplots(figsize=(18,9))
        plt.ylabel("nb demandes d'intervention",fontsize=14)
        plt.xlabel("pas par date",fontsize=14)
        plt.title('prediction et valeurs réelles : ('+str(nb_elmnts_to_print)+' premiers éléments)')
        #plotting in truth : tests values 
        ax.plot(y_test[:nb_elmnts_to_print],'-o', color="blue", label="réel nbDi")
        ax.plot(test_predict[:nb_elmnts_to_print],'-o', color="green",label="prédiction nbDi")
        plt.legend()
        plt.savefig(pathOutput+'3- predictOnTest.png')
    finally:
        plt.close()

'''
predict feature : (nb_days_predict) days
'''
def graphPredictNextDays(last_data, NB_DAYS_PREDICTED, dijon, feature, fromDateToNumberAfter, dijon_timestamps, dijon_dates):
    try:
        plt.subplots(figsize=(24,11))
        plt.xticks(rotation=90)
        plt.xlabel("date de demande d'intervention",fontsize=14)
        plt.ylabel("nombre de demande d'intervention",fontsize=14)
        plt.title(str(last_data)+' derniers jours + ' + 'prédiction nbDi par date ('+ str(NB_DAYS_PREDICTED)+' futur jours)')
        #getting 2 last month values for
        dijon_labels = np.array(pd.DataFrame(dijon['nbDi'], dtype='float').tail(last_data)).flatten()
        #plotting features values
        plt.plot(*zip(*sorted(zip(dijon_timestamps,dijon_labels))), color='blue', label='truth ' + str('in 62 last days'))
        plt.plot(*zip(*sorted(zip(dijon_dates, feature))), 'b:o', color='blue', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
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
        plt.bar(dijon_dates, feature, color='red', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
        plt.title(str(NB_DAYS_PREDICTED) + ' next predictions values (bar chart)')
        plt.xlabel("date de demande d'intervention",fontsize=14)
        plt.ylabel("nombre de demande d'intervention",fontsize=14)
        plt.legend()
        plt.savefig(pathOutput+'5- predict_'+str(NB_DAYS_PREDICTED)+'_next_days.png')
    finally:
        plt.close()