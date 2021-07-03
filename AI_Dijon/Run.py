import os,sys
sys.path.insert(1, os.path.abspath('.'))
import numpy as np
import pandas as pd
from datetime import *
from keras.callbacks import EarlyStopping
from sklearn import *
from Preprocessing import *
from Prediction import *
from evaluate.ErrorsPrediction import *
from BuildingModel import * 
from DataAugmentate import *
from DataSetForModel import *
from future.MeteoFuture import *
from RequestFromDataBase import *
from evaluate.EvaluateModel import *
from generate_graph.GenerateGraph import *

#number of time steps
LOOK_BACK = 7
NB_DAYS_PREDICTED, UNITS, EPOCHS, BATCH_SIZE, TEST_SIZE, SHUFFLE = 15, 150, 500, 32, 0.2, False
rootOutputFile= "files/output/"
'''
generating folders root path
'''
if not os.path.exists(rootOutputFile):os.makedirs(rootOutputFile)
dateNbDiTupleArray, dateMeteoTupleArray = requestDiByDate(), requestMeteoByDate()
addingHidenDay=matchingDateStartEnd(dateBetweenStartEnd(dateNbDiTupleArray)[2], dict(dateNbDiTupleArray))
print("start analysing from {} to {} : the last date in the potgresql bdd\ntotal number of days : {} days".format(
dateBetweenStartEnd(dateNbDiTupleArray)[0], dateBetweenStartEnd(dateNbDiTupleArray)[1], len(addingHidenDay)))
print("number of days in the period interventions are requested : {} days".format(len(dateNbDiTupleArray)))
df=pd.DataFrame(addingHidenDay, columns=['date','nbDi'])
#adding frequence, peak and boolean request to the dataframe using nbDi
df['freq_nbDi'], df['is_peak_nbDi'], df['is_request_nbDi'] = freq_nbDi(addingHidenDay), is_peak_nbDi(addingHidenDay), is_request_nbDi(addingHidenDay)
'''
adding meteo datas to the dataframe
'''
df['mto_temp(celcius)'], df['mto_temp_min(celcius)'] = [i[1] for i in dateMeteoTupleArray], [i[2] for i in dateMeteoTupleArray]
df['mto_temp_max(celcius)'], df['mto_pressure(hPa)']= [i[3] for i in dateMeteoTupleArray], [i[4] for i in dateMeteoTupleArray]
df['mto_humidity(%)'], df['mto_visibility(km)'] = [i[5] for i in dateMeteoTupleArray], [i[6] for i in dateMeteoTupleArray]
df['mto_wind_speed(m s)'], df['mto_clouds(%)'] = [i[7] for i in dateMeteoTupleArray], [i[8] for i in dateMeteoTupleArray]
'''
generating future meteo files
'''
meteoFuture()
futureMeteo = loadCsvFile("files/future/generated/CSV/futureMeteoByDate.csv")
####################################### GENERATING OUPUT FILES #####################################################################
'''
saving in imgs folder the graphs nbDi or meteo by date
'''
graphNbDiMeteoByDate(df)
'''
creating the new dijon dataframe
'''
'''dijon=df[["nbDi", "freq_nbDi", "is_peak_nbDi", "is_request_nbDi",'mto_temp(celcius)' ,'mto_temp_min(celcius)',
'mto_temp_max(celcius)','mto_pressure(hPa)', 'mto_humidity(%)', 'mto_visibility(km)', 'mto_wind_speed(m s)', 
'mto_clouds(%)']].astype('float')'''
dijon=df[["nbDi"]].astype('float')
'''
creating the dataset
'''
df_scaled, X, Y = create_lstm_dataset(dijon, LOOK_BACK)
'''
writing output files -- dijon transformed file
'''
try:
    file1, file2, file3 = open(rootOutputFile+"1- init_dataframe.txt", "w"), open(rootOutputFile+"3- X_dataset.txt", "w"),open(rootOutputFile+"4- Y_dataset.txt", "w")
    file1.write(str(dijon.head(50)))
    file2.write(str(X.shape)+'\n'+str(X[0:50]))
    file3.write(str(Y.shape)+'\n'+str(Y[-50:]))
finally:
    for i in [file1, file2, file3]:i.close()
'''
spliting data_set
'''
dijon_train, dijon_test, label_train, label_test=model_selection.train_test_split(X, Y, test_size=TEST_SIZE, shuffle=SHUFFLE)
'''
writting in the output files
'''
try:
    
    file1, file2 = open(rootOutputFile+"5- dijon_train.txt", "w"),open(rootOutputFile+"6- label_train.txt", "w")
    file3, file4 = open(rootOutputFile+"7- dijon_test.txt", "w"),open(rootOutputFile+"8- label_test.txt", "w")
    file1.write(str(dijon_train.shape)+'\n'+str(dijon_train[0:50])) 
    file2.write(str(label_train.shape)+'\n'+str(label_train[0:50]))
    file3.write(str(dijon_test.shape)+'\n'+str(dijon_test[0:50]))
    file4.write(str(label_test.shape)+'\n'+str(label_test[0:50]))
finally:
    for i in [file1, file2, file3, file4]:i.close()
'''
EarlyStopping to prevent the overfitting on the losses and building the RNN model
'''
es, model = EarlyStopping(monitor='val_loss', patience=6), buildModel(UNITS, dijon, LOOK_BACK)
history = model.fit(dijon_train, label_train, verbose=2, validation_split=0.2, epochs=EPOCHS, shuffle=False,
 batch_size=BATCH_SIZE, callbacks=[es])
'''
getting the RNN model result
'''
model_eval = evaluateModel(model, dijon_train, label_train, dijon_test, label_test)
print(model_eval)
'''
saving the graph of model history
'''
graphHistoryModel(history)
####################################### PREDICTIONS #####################################################################
#treatment tests values
label_test, nb_elmnts_to_print = np.repeat(label_test, dijon.shape[1], axis=-1), 30
y_test = (df_scaled.inverse_transform(label_test)[:,0]).reshape(label_test.shape[0], 1)
#test predict values
test_predict=model.predict(dijon_test, batch_size=32, verbose = 2)
test_predict = np.repeat(test_predict, dijon.shape[1], axis=-1)
test_predict = (df_scaled.inverse_transform(test_predict)[:,0]).reshape(test_predict.shape[0], 1)
print('nb elements in test :',len(y_test) , '\nnb elements to plot :', nb_elmnts_to_print, 'premiers test éléments')
'''
plotting truth and nbDi prediction
'''
graphTruthOnPrediction(nb_elmnts_to_print, y_test, test_predict)
'''
mesuring performances of the model
'''
generating_errors(y_test, test_predict, nb_elmnts_to_print)
'''
predict feature : (nb_days_predict) days
'''
feature, LAST_NB_DATA = np.rint(predictNextDays(model, dijon, NB_DAYS_PREDICTED, df_scaled, LOOK_BACK, futureMeteo)), 62
dijon_timestamps = np.array(pd.DataFrame(df[["date"]]).tail(LAST_NB_DATA)).flatten()
fromDateToNumberAfter = listDatesBetweenDateAndNumber(date.fromisoformat(dijon_timestamps[len(dijon_timestamps)-1]), NB_DAYS_PREDICTED)
dijon_dates = np.array(fromDateToNumberAfter, dtype='datetime64[D]').astype(str)
graphPredictNextDays(LAST_NB_DATA, NB_DAYS_PREDICTED, dijon, feature, fromDateToNumberAfter, dijon_timestamps, dijon_dates)
'''
graph with just feature informations
'''
graphFeatureInfos(dijon_dates, feature, NB_DAYS_PREDICTED)