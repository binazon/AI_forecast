#######################################AUTORS#############################################################################
# Main file
#######################################IMPORTING###########################################################################
import math
from numbers import Number
import sys, os
from typing import List
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import csv_manager
import prediction
from errors_prediction import generating_errors
from numpy import argmax
from datetime import *
from tensorflow import keras
from tensorflow.keras import *
from keras import Input, Model
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.feature_selection import *
import seaborn as sns
#######################################VARIABLES#####################################################################
scaler0To1 = MinMaxScaler(feature_range=(-1,1))
#number of time steps
look_back = 3
#number days trainna to predict
nb_days_predict = 15
UNITS = 150
epochs = 1500
batch_size =32
test_size = 0.2
#######################################METHODS#####################################################################
#creating the lstm dataset
def create_lstm_dataset(df, look):
    number_of_rows = df.shape[0]
    number_of_features = df.shape[1]
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data = transformed_dataset, index=df.index)
    #writing the dijon trnsformed file
    f = open("files/2- dijon_df_transformed.txt", "w")
    f.write(str(transformed_df.head(50)))
    f.close()
    train = np.empty([number_of_rows - look, look, number_of_features], dtype='float')
    label = np.empty([number_of_rows - look, number_of_features])
    for i in range(0, number_of_rows-look):
        train[i] = transformed_df.iloc[i:i+look, 0:number_of_features]
        label[i] = transformed_df.iloc[i+look:i+look+1, 0:number_of_features]
    return train, label, scaler
#using data preprocessing - simple LSTM RNN Model
# -- each next value is based on look_back previous values
def prepare_data(timeseries_data, look_back):
    X, y = [],[]
    for i in range(len(timeseries_data)):
        end_ix = i + look_back
        if(end_ix >= len(timeseries_data)):
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x.flatten())
        y.append(seq_y[0])
    return np.array(X), np.array(y)
#triying to build the model
def buildModel(enter, data):
    model = Sequential()
    model.add(LSTM(55, dropout= 0.02, batch_input_shape=(None,data.shape[1],data.shape[2]), return_sequences=True))
    model.add(LSTM(40, dropout=0.02, activity_regularizer=regularizers.l2(1e-5), return_sequences=False))
    model.add(Dense(data.shape[2]))
    model.compile(optimizer='adam', loss='mse', metrics=['acc', 'mae'])
    return model
#date bettween start en end of all DI - even date not in bdd. 
def dateBetweenStartEnd(_array) -> List:
    return _array[0][1],_array[len(_array)-1][1], np.array(pd.date_range(_array[0][1],_array[len(_array)-1][1]))
#get list of date between date and nb following days
def listDatesBetweenDateAndNumber(date, number) -> List:
    start = date + timedelta(days=1)
    end = date + timedelta(days=number)
    return np.array(pd.date_range(start, end))
#normalising dataArray with values between 0 and 1
def normalisingArray(dataArray) -> List:
    return scaler0To1.fit_transform(dataArray)
#removing normalisation of dataArray : from values between 0 and 1 to real values
def removing_normalingArray(dataArray) -> List:
    return scaler0To1.inverse_transform(dataArray)
#getting frequence of nbDi in nbDi column
def freq_nbDi(data) -> List:
    tab = []
    pos = 1
    for i in data[:,1]:
        tab.append(data[:,1][0:pos].tolist().count(i))
        pos = pos + 1
    return tab
#getting all peak all nbDi
def is_peak_nbDi(data) -> List:
    peak = []
    for i in data[:,1]:
        if(int(i) >= 6):
            peak.append(1)
        else:
            peak.append(0)
    return peak
#getting all ways without request of intervention
def is_request_nbDi(data) -> List : 
    request = []
    for i in data[:,1]:
        if(int(i) > 0):
            request.append(1)
        else:
            request.append(0)
    return request
#######################################RUNNING#######################################################################
data_array=csv_manager.sortDijonExtractByDate(csv_manager.loadCsvFile('database/dijonData_extract_19_04_2021.csv'))
meteo = csv_manager.loadCsvFile('database/meteo_07_03_2019_to_30_04_2021.csv')
print("nb line in dijon database --csv file-- : {}".format(len(data_array)))
data_array = csv_manager.datetime_to_date(data_array)
csv_manager.groupByDateAndComments(csv_manager.saltingComments(data_array))
groupByDateAndNbDi=csv_manager.groupingByDateAndDI(data_array)
nbDiByDateArray=np.array(list(groupByDateAndNbDi.items()))
print("nb line/distinct date in dijon database base : {}".format(len(nbDiByDateArray)))
print('starting date in bdd : ' + str(dateBetweenStartEnd(data_array)[0]) + ', ending date : ' + str(dateBetweenStartEnd(data_array)[1]))
evenHidenDateDijonBDD=csv_manager.matchingDateStartEnd(dateBetweenStartEnd(data_array)[2], groupByDateAndNbDi)
print("nb line/distinct date after matching all days : {}".format(len(evenHidenDateDijonBDD)))
df=pd.DataFrame(evenHidenDateDijonBDD, columns=['date','nbDi'])
#adding some informations to our dataframe
df['freq_nbDi'] = freq_nbDi(evenHidenDateDijonBDD)
df['is_peak_nbDi'] = is_peak_nbDi(evenHidenDateDijonBDD)
df['is_request_nbDi'] = is_request_nbDi(evenHidenDateDijonBDD)
#adding meteo informations to our dataframe
df['vitesse_vent_max'] = meteo['VITESSE_VENT_MAX_KMH']
df['couverture_nuageuse'] = meteo['COUVERTURE_NUAGEUSE_MOYENNE_PERCENT']
df['visibilitee'] = meteo['VISIBILITE_MOYENNE_KM']
df['temp_day'] = meteo[["TEMPERATURE_MATIN_C","TEMPERATURE_MIDI_C","TEMPERATURE_SOIREE_C"]].mean(axis=1)
df['min_temp_c'] = meteo['MIN_TEMPERATURE_C']
df['max_temp_c'] = meteo['MAX_TEMPERATURE_C']
df['pression'] = meteo['PRESSION_MAX_MB']
df['humiditee_max'] = meteo['HUMIDITE_MAX_POURCENT']
#############################################################################
dijon_timestamps=df[["date"]]
#plotting and saving all nbDi by date
plt.subplots(figsize=(24,11))
plt.plot(*zip(*sorted(zip(dijon_timestamps.values.flatten(),df["nbDi"].astype(int)))), color='blue', label='nbDi')
plt.xticks(df.index, dijon_timestamps.values.flatten(), rotation=90)
plt.locator_params(axis='x', nbins=15)
plt.ylabel("nbDi",fontsize=14)
plt.xlabel("date",fontsize=14)
plt.legend()
plt.savefig('imgs/1- nbDiByDate.png')
plt.close()
#extends nbDi data with equivalent datas : freq_nbDi, is_peak, ...
dijon=df[["nbDi", "freq_nbDi", "is_peak_nbDi", "is_request_nbDi"]].astype('float')
#writing the dijon trnsformed file
f1, f2, f3 = open("files/1- init_dataframe.txt", "w"), open("files/3- X_dataset.txt", "w"),open("files/4- Y_dataset.txt", "w")
f1.write(str(dijon.head(50)))
X, Y, df_scaled = create_lstm_dataset(dijon, look_back)
# writing dataset in file 
f2.write(str(X.shape)+'\n'+str(X[0:50])) 
f3.write(str(Y.shape)+'\n'+str(Y[0:50]))
f1.close()
f2.close()
f3.close()
#spliting data_set
dijon_train, dijon_test, label_train, label_test=train_test_split(X, Y, test_size=test_size, shuffle=False)
#writing the train and tests values
f1, f2, f3, f4 = open("files/5- dijon_train.txt", "w"),open("files/6- label_train.txt", "w"),open("files/7- dijon_test.txt", "w"),open("files/8- label_test.txt", "w")
f1.write(str(dijon_train.shape)+'\n'+str(dijon_train[0:50])) 
f2.write(str(label_train.shape)+'\n'+str(label_train[0:50]))
f3.write(str(dijon_test.shape)+'\n'+str(dijon_test[0:50]))
f4.write(str(label_test.shape)+'\n'+str(label_test[0:50]))
f1.close()
f2.close()
f3.close()
f4.close()
#EarlyStopping to prevent the overfitting on the losses
es = EarlyStopping(monitor='val_loss', patience=3)
#building the model LSTM - Long Short Time Memory
model = buildModel(UNITS, dijon_train)
#20% of validation data are used on the train dataset
history = model.fit(dijon_train, label_train, validation_data=(dijon_test, label_test), epochs=epochs, batch_size=batch_size, callbacks=[es])
#evaluation in train dataset
eval_train = model.evaluate(dijon_train, label_train)
print("taux de pertes -- train :",eval_train[0]*100 , "%")
print("accuracy -- train :",eval_train[1]*100 , "%")
print("erreure absolue moyenne -- train :",eval_train[2]*100 , "%")
#evaluation in test dataset
eval_test = model.evaluate(dijon_test, label_test)
print("taux de pertes -- test :",eval_test[0]*100 , "%")
print("accuracy --test :",eval_test[1]*100 , "%")
print("erreure absolue moyenne --test :",eval_test[2]*100 , "%")

'''val_score = []
for k in range(1,100):
    score = cross_val_score(model, dijon_train, label_test, cv=5).mean()
    val_score.append(score)
plt.plot(val_score)
plt.savefig('imgs/toto.png')
plt.close()'''

#############################################################################
plt.subplots(figsize=(18,9))
plt.plot(history.history['loss'], label = 'train_losses')
plt.plot(history.history['acc'], label = 'train_accuracy')
plt.plot(history.history['val_loss'], label='val_losses')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel("nb epochs",fontsize=14)
plt.ylabel("accuracy and losses",fontsize=14)
plt.legend()
plt.savefig('imgs/2- history_train.png')
plt.close()
#############################################################################
#treatment tests values
#df_scaled = StandardScaler().fit(label_test)
y_test = df_scaled.inverse_transform(label_test)
#test predict values
    ##removing normaling to plot graph
test_predict=model.predict(dijon_test, batch_size=32, verbose = 2)
test_predict = df_scaled.inverse_transform(test_predict)
nb_elmnts_to_print = 30
print('nb elements in test :',len(y_test))
print('nb elements to plot :', nb_elmnts_to_print, 'premiers test éléments')
#plotting truth and nbDi prediction
fig,ax=plt.subplots(figsize=(18,9))
plt.xticks(rotation=90)
plt.ylabel("nb demandes d'intervention",fontsize=14)
plt.xlabel("pas par date",fontsize=14)
plt.title('prediction et valeurs réelles : ('+str(nb_elmnts_to_print)+' premiers éléments)')
#printing first elements
ax.plot(y_test[:,0][:nb_elmnts_to_print],'-o', color="blue", label="réel nbDi")
ax.plot(test_predict[:,0][:nb_elmnts_to_print],'-o', color="green",label="prédiction nbDi")
plt.legend()
plt.savefig('imgs/3- predictOnTest.png')
plt.close()
#############################################################################
#mesuring performances of the model
generating_errors(y_test, test_predict, nb_elmnts_to_print)
#############################################################################
#predict feature : (nb_days_predict) days
feature = np.rint(prediction.predictNextDays(model, dijon, nb_days_predict, df_scaled, look_back))
#############################################################################
last_data = 62
plt.subplots(figsize=(18,9))
plt.xticks(rotation=90)
plt.xlabel("date de demande d'intervention",fontsize=14)
plt.ylabel("nombre de demande d'intervention",fontsize=14)
plt.title(str(last_data)+' derniers jours + ' + 'prédiction nbDi par date ('+ str(nb_days_predict)+' futur jours)')
#getting 2 last month values for 
dijon_timestamps = np.array(pd.DataFrame(dijon_timestamps).tail(last_data)).flatten()
dijon_labels = np.array(pd.DataFrame(dijon['nbDi'], dtype='float').tail(last_data)).flatten()
#plotting features values
dijon_dates = listDatesBetweenDateAndNumber(date.fromisoformat(dijon_timestamps[len(dijon_timestamps)-1]), nb_days_predict)
dijon_dates = [str(dijon_dates[i]).split("T")[0] for i in range(len(dijon_dates))]
plt.plot(*zip(*sorted(zip(dijon_timestamps,dijon_labels))), color='blue', label='truth ' + str('in 62 last days'))
plt.plot(*zip(*sorted(zip(dijon_dates, feature))), 'b:o', color='blue', label=str(nb_days_predict) + ' feature(s)')
plt.legend()
plt.savefig('imgs/4- last '+str(last_data)+' jours + predict_'+str(nb_days_predict)+'_next_days.png')
plt.close()
#############################################################################

'''#f-score selection calculate
selector = SelectKBest(f_classif, k=10)
selected_features = selector.fit_transform(dijon_dates, label_train)
'''

plt.subplots(figsize=(24,11))
plt.xticks(rotation=90)
plt.bar(dijon_dates, feature, color='red', label=str(nb_days_predict) + ' feature(s)')
plt.title(str(nb_days_predict) + ' next predictions values (bar chart)')
plt.xlabel("date de demande d'intervention",fontsize=14)
plt.ylabel("nombre de demande d'intervention",fontsize=14)
plt.legend()
plt.savefig('imgs/5- predict_'+str(nb_days_predict)+'_next_days.png')
plt.close()
#############################################################################
print(pd.DataFrame({'Next dates':dijon_dates,'nbDi output':feature}))