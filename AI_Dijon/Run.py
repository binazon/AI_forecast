import os,sys
sys.path.insert(1, os.path.abspath('.'))
import numpy as np
import pandas as pd
from datetime import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import *
from Preprocessing import *
from Prediction import *
from evaluate.ErrorsPrediction import *
from BuildingModel import * 
from DataSetForModel import *
from future.MeteoFuture import *
from RequestFromDataBase import *
from evaluate.EvaluateModel import *
from generate_graph.Graph import *
from CrossValidation import *

'''
timesteps or lookback : less or equal to 5
openweathermap API get us 5 last meteo history from current day
'''
LOOK_BACK = 7
NB_DAYS_PREDICTED, EPOCHS, BATCH_SIZE, TEST_SIZE, VALIDATION_SPLIT, DROPOUT, WEIGHT_CONSTRAINT, SHUFFLE = 7, 500, 30, 0.2, 0.2, 0.1, 0, True
PATIENCE = 60
rootOutputFile= "generated/files/"
'''
generating folders root path
'''
if not os.path.exists(rootOutputFile):os.makedirs(rootOutputFile)
dateNbDiTupleArray = requestDiByDate()
addingHidenDay=matchingDateStartEnd(dateBetweenStartEnd(dateNbDiTupleArray)[2], dict(dateNbDiTupleArray))
print("start the analyse from {} to {} : the last date in the potgresql bdd\ntotal number of days : {} days".format(
dateBetweenStartEnd(dateNbDiTupleArray)[0], dateBetweenStartEnd(dateNbDiTupleArray)[1], len(addingHidenDay)))
print("number of days in the period interventions are requested : {} days".format(len(dateNbDiTupleArray)))
df=pd.DataFrame(addingHidenDay, columns=['date','nbDi'])

dateMeteoTupleArray = requestMeteoByDate(len(addingHidenDay))
'''
adding meteo datas to the dataframe
'''
df['mto_temp(celcius)'], df['mto_temp_min(celcius)'] = [i[1] for i in dateMeteoTupleArray], [i[2] for i in dateMeteoTupleArray]
df['mto_temp_max(celcius)'], df['mto_pressure(hPa)']= [i[3] for i in dateMeteoTupleArray], [i[4] for i in dateMeteoTupleArray]
df['mto_humidity(%)'], df['mto_visibility(km)'] = [i[5] for i in dateMeteoTupleArray], [i[6] for i in dateMeteoTupleArray]
df['mto_wind_speed(m s)'], df['mto_clouds(%)'] = [i[7] for i in dateMeteoTupleArray], [i[8] for i in dateMeteoTupleArray]
'''
generating future meteo files and load CSV future_meteo file 
'''
meteoFuture()
futureMeteo = loadCsvFile("generated/future/CSV/futureMeteoByDate.csv")
####################################### GENERATING OUPUT FILES #####################################################################
'''
saving in graph folder the graphs nbDi or meteo by date
'''
graphNbDiMeteoByDate(df)
'''
creating the new dijon dataframe
'''
dijon=df[["nbDi", 'mto_temp(celcius)' ,'mto_temp_min(celcius)',
'mto_temp_max(celcius)','mto_pressure(hPa)', 'mto_humidity(%)', 'mto_visibility(km)', 'mto_wind_speed(m s)', 
'mto_clouds(%)']].astype('float')
#dijon=df[["nbDi"]].astype('float')
'''
creating the dataset
'''
X, Y = create_lstm_dataset(dijon, LOOK_BACK)
'''
building the RNN model
'''
model = buildModel(X.shape[1], DROPOUT, WEIGHT_CONSTRAINT, X, Y)
'''
writing output files -- dijon transformed file
'''
try:
    file1, file2, file3 = open(rootOutputFile+"1- init_dataframe.txt", "w+"), open(rootOutputFile+"3- X_dataset.txt", "w+"),open(rootOutputFile+"4- Y_dataset.txt", "w+")
    file1.write(str(dijon.head(50)))
    file2.write(str(X.shape)+'\n'+str(X[0:50]))
    file3.write(str(Y.shape)+'\n'+str(Y[-50:]))
except OSError as error:
    print("cannot not open or write in file", error)
finally:
    for i in [file1, file2, file3]:i.close()
'''
spliting data_set
'''
dijon_train, dijon_test, label_train, label_test=model_selection.train_test_split(X, Y, test_size=TEST_SIZE, shuffle=SHUFFLE)
'''
searching good hyperparameters for the model
hyperparams generated there are replaced in the model
'''
print(fixHyperParamsGridSearch(buildModel, dijon_train, label_train))
'''
writting in the output files
'''
try:
    file1, file2 = open(rootOutputFile+"5- dijon_train.txt", "w+"),open(rootOutputFile+"6- label_train.txt", "w+")
    file3, file4 = open(rootOutputFile+"7- dijon_test.txt", "w+"),open(rootOutputFile+"8- label_test.txt", "w+")
    file1.write(str(dijon_train.shape)+'\n'+str(dijon_train[0:50])) 
    file2.write(str(label_train.shape)+'\n'+str(label_train[0:50]))
    file3.write(str(dijon_test.shape)+'\n'+str(dijon_test[0:50]))
    file4.write(str(label_test.shape)+'\n'+str(label_test[0:50]))
except OSError as error:
    print("cannot not open or write in file", error)
finally:
    for i in [file1, file2, file3, file4]:i.close()
'''
EarlyStopping to prevent the overfitting on the losses
'''
es= EarlyStopping(monitor='val_loss', verbose=1, patience=PATIENCE), 
history = model.fit(dijon_train, label_train, verbose=1, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, shuffle=SHUFFLE,
 batch_size=BATCH_SIZE, callbacks=[es])
'''
getting the RNN model result
'''
print(evaluateModel(model, dijon_train, label_train, dijon_test, label_test))
'''
saving the graph of model history
'''
graphHistoryModel(history)
####################################### PREDICTIONS #####################################################################
'''
treatment tests values
'''
label_test, nb_elmnts_to_print = np.repeat(label_test, dijon.shape[1], axis=-1), 50
y_test = (unormaliseData(label_test)[:,0]).reshape(label_test.shape[0], 1)
'''
testing predicted values
'''
test_predict=model.predict(dijon_test, batch_size=32, verbose = 1)
test_predict = np.repeat(test_predict, dijon.shape[1], axis=-1)
test_predict = (unormaliseData(test_predict)[:,0]).reshape(test_predict.shape[0], 1)
print('nb elements in test :',len(y_test))
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
feature, LAST_NB_DATA = np.rint(predictNextDays(model, dijon, NB_DAYS_PREDICTED, LOOK_BACK, futureMeteo)), 62
dijon_timestamps = np.array(pd.DataFrame(df[["date"]]).tail(LAST_NB_DATA)).flatten()
fromDateToNumberAfter = listDatesBetweenDateAndNumber(date.fromisoformat(
    dijon_timestamps[len(dijon_timestamps)-1]), NB_DAYS_PREDICTED)
dijon_dates = np.array(fromDateToNumberAfter, dtype='datetime64[D]').astype(str)
graphPredictNextDays(LAST_NB_DATA, NB_DAYS_PREDICTED, dijon, feature, dijon_timestamps, dijon_dates)
'''
graph with just feature informations
'''
graphFeatureInfos(dijon_dates, feature, NB_DAYS_PREDICTED)