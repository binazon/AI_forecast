#######################################AUTORS#############################################################################
# Main file
#######################################IMPORTING###########################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import *
from Prediction import *
from ErrorsPrediction import *
from BuildingModel import * 
from DataAugmentate import *
from DataSetForModel import *
from MeteoFuture import *
from RequestFromDataBase import *
from numpy import argmax
from datetime import *
from tensorflow.keras import *
from keras import Input, Model
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.feature_selection import *
#######################################VARIABLES#####################################################################
scaler0To1 = MinMaxScaler(feature_range=(-1,1))
#number of time steps
look_back = 14
nb_days_predict, UNITS, epochs, batch_size, test_size = 15, 150, 500, 32, 0.2
#######################################RUNNING#######################################################################
dateNbDiTupleArray, dateMeteoTupleArray = requestDiByDate(), requestMeteoByDate()
print("nb line/distinct date in dijon database base : {}\nstart analysing from : {}, to : {} last date in the database".format(len(dateNbDiTupleArray), str(dateBetweenStartEnd(dateNbDiTupleArray)[0])), str(dateBetweenStartEnd(dateNbDiTupleArray)[1]))
evenHidenDay=matchingDateStartEnd(dateBetweenStartEnd(dateNbDiTupleArray)[2], dict(dateNbDiTupleArray))
print("nb line/distinct date after matching days without intervention : {}".format(len(evenHidenDay)))
#creating the first data frame
df=pd.DataFrame(evenHidenDay, columns=['date','nbDi'])
#adding some informations to our dataframe
df['freq_nbDi'] = freq_nbDi(evenHidenDay)
df['is_peak_nbDi'] = is_peak_nbDi(evenHidenDay)
df['is_request_nbDi'] = is_request_nbDi(evenHidenDay)
#adding meteo informations to our dataframe
df['mto_temp(celcius)'] = [i[1] for i in dateMeteoTupleArray]
df['mto_temp_min(celcius)'] = [i[2] for i in dateMeteoTupleArray]
df['mto_temp_max(celcius)'] = [i[3] for i in dateMeteoTupleArray]
df['mto_pressure(hPa)'] = [i[4] for i in dateMeteoTupleArray]
df['mto_humidity(%)'] = [i[5] for i in dateMeteoTupleArray]
df['mto_visibility(km)'] = [i[6] for i in dateMeteoTupleArray]
df['mto_wind_speed(m/s)'] = [i[7] for i in dateMeteoTupleArray]
df['mto_clouds(%)'] = [i[8] for i in dateMeteoTupleArray]
#############################################################################
dijon_timestamps=df[["date"]]
#plotting and saving all nbDi and meteo datas by date
for i in range(1, len(df.columns)):
    plt.subplots(figsize=(24,11))
    plt.plot(*zip(*sorted(zip(dijon_timestamps.values.flatten(),df[df.columns[i]].astype(float)))), 
    color='blue', label=df.columns[i])
    plt.xticks(df.index, dijon_timestamps.values.flatten(), rotation=90)
    plt.locator_params(axis='x', nbins=15)
    plt.xlabel("date",fontsize=14)
    plt.ylabel(df.columns[i],fontsize=14)
    plt.legend()
    plt.savefig('imgs/input/1.'+str(i)+ '- '+df.columns[i]+'ByDate.png')
    plt.close()
#extends nbDi data with augmented datas : freq_nbDi, is_peak, ...
dijon=df[["nbDi", "freq_nbDi", "is_peak_nbDi", "is_request_nbDi",'mto_temp(celcius)' ,'mto_temp_min(celcius)' ,
'mto_temp_max(celcius)','mto_pressure(hPa)', 'mto_humidity(%)', 'mto_visibility(km)', 'mto_wind_speed(m/s)', 
'mto_clouds(%)']].astype('float')
#writing the dijon trnsformed file
f1, f2, f3 = open("files/1- init_dataframe.txt", "w"), open("files/3- X_dataset.txt", "w"),open("files/4- Y_dataset.txt", "w")
f1.write(str(dijon.head(50)))
X, Y, df_scaled = create_lstm_dataset(dijon, look_back)
# writing dataset in file 
f2.write(str(X.shape)+'\n'+str(X[0:50]))
f3.write(str(Y.shape)+'\n'+str(Y[-50:]))
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
es = EarlyStopping(monitor='val_loss', patience=6)
#building the model LSTM - Long Short Time Memory
model = buildModel(UNITS, dijon_train, label_train)
#20% of validation data are used on the train dataset
#history = model.fit(dijon_train, label_train, verbose=2, validation_split=0.2, epochs=epochs, shuffle=False, batch_size=batch_size, callbacks=[es])
history = model.fit(dijon_train, label_train, verbose=2, validation_split=0.2, epochs=epochs, 
shuffle=False, batch_size=batch_size)
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
#############################################################################
plt.subplots(figsize=(18,9))
plt.plot(history.history['loss'], label = 'train_losses')
plt.plot(history.history['acc'], label = 'train_accuracy')
plt.plot(history.history['val_loss'], label='val_losses')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel("epoch",fontsize=14)
plt.ylabel("validation and train values",fontsize=14)
plt.legend()
plt.savefig('imgs/output/2- history_train.png')
plt.close()
#############################################################################
#treatment tests values
label_test = np.repeat(label_test, dijon.shape[1], axis=-1)
y_test = (df_scaled.inverse_transform(label_test)[:,0]).reshape(label_test.shape[0], 1)
#test predict values
    ##removing normaling to plot graph
test_predict=model.predict(dijon_test, batch_size=32, verbose = 2)
test_predict = np.repeat(test_predict, dijon.shape[1], axis=-1)
test_predict = (df_scaled.inverse_transform(test_predict)[:,0]).reshape(test_predict.shape[0], 1)
nb_elmnts_to_print = 30
print('nb elements in test :',len(y_test))
print('nb elements to plot :', nb_elmnts_to_print, 'premiers test éléments')
#plotting truth and nbDi prediction
fig,ax=plt.subplots(figsize=(18,9))
plt.ylabel("nb demandes d'intervention",fontsize=14)
plt.xlabel("pas par date",fontsize=14)
plt.title('prediction et valeurs réelles : ('+str(nb_elmnts_to_print)+' premiers éléments)')
#plotting in truth : tests values 
ax.plot(y_test[:nb_elmnts_to_print],'-o', color="blue", label="réel nbDi")
ax.plot(test_predict[:nb_elmnts_to_print],'-o', color="green",label="prédiction nbDi")
plt.legend()
plt.savefig('imgs/output/3- predictOnTest.png')
plt.close()
#############################################################################
#mesuring performances of the model
generating_errors(y_test, test_predict, nb_elmnts_to_print)
#############################################################################
#predict feature : (nb_days_predict) days
feature = np.rint(predictNextDays(model, dijon, nb_days_predict, df_scaled, look_back))
#############################################################################
last_data = 62
plt.subplots(figsize=(24,11))
plt.xticks(rotation=90)
plt.xlabel("date de demande d'intervention",fontsize=14)
plt.ylabel("nombre de demande d'intervention",fontsize=14)
plt.title(str(last_data)+' derniers jours + ' + 'prédiction nbDi par date ('+ str(nb_days_predict)+' futur jours)')
#getting 2 last month values for 
dijon_timestamps = np.array(pd.DataFrame(dijon_timestamps).tail(last_data)).flatten()
dijon_labels = np.array(pd.DataFrame(dijon['nbDi'], dtype='float').tail(last_data)).flatten()
#plotting features values
dijon_dates = np.array(listDatesBetweenDateAndNumber(date.fromisoformat(dijon_timestamps[len(dijon_timestamps)-1]), 
nb_days_predict), dtype='datetime64[D]').astype(str)
plt.plot(*zip(*sorted(zip(dijon_timestamps,dijon_labels))), color='blue', label='truth ' + str('in 62 last days'))
plt.plot(*zip(*sorted(zip(dijon_dates, feature))), 'b:o', color='blue', label=str(nb_days_predict) + ' feature(s)')
plt.legend()
plt.savefig('imgs/output/4- last '+str(last_data)+' jours + predict_'+str(nb_days_predict)+'_next_days.png')
plt.close()
#############################################################################
plt.subplots(figsize=(24,11))
plt.xticks(rotation=90)
plt.bar(dijon_dates, feature, color='red', label=str(nb_days_predict) + ' feature(s)')
plt.title(str(nb_days_predict) + ' next predictions values (bar chart)')
plt.xlabel("date de demande d'intervention",fontsize=14)
plt.ylabel("nombre de demande d'intervention",fontsize=14)
plt.legend()
plt.savefig('imgs/output/5- predict_'+str(nb_days_predict)+'_next_days.png')
plt.close()
#############################################################################
print(pd.DataFrame({'Next dates':dijon_dates,'nbDi output':feature}))