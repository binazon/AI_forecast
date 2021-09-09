import os,sys

from tensorflow.python.keras.saving.save import load_model
sys.path.insert(1, os.path.abspath('.'))
import numpy as np
import pandas as pd
import Globals.Variable as global_vars
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import *
from Preprocess import *
from Prediction import *
from Evaluate.Evaluate import *
from Model import * 
from Future.MeteoFuture import *
from Graph.Graph import *
from GridSearchCV import *
from DatabaseConnectivity.RequestFromDataBase import *
from DatabaseConnectivity.ConnectPostgreDataBase import *
from imblearn.over_sampling import SMOTE

'''
class used to run the prediction project
'''
class Run :
    def main():
        '''
        timestamp or look_back (this value need to stay lesser than 3 weeks in timeseries analysis)
        '''
        LOOK_BACK = 7
        NB_DAYS_PREDICTED, EPOCHS, BATCH_SIZE, TEST_SIZE, VALIDATION_SPLIT, DROPOUT, WEIGHT_CONSTRAINT = 7,1000, 20, 0.1, 0.2, 0.1, 1
        OPTIMIZER, PATIENCE, rootOutputFile, rootOutputModel = 'Adam', 20, "generated/files/", "generated/model/"
        '''
        generating folders root path
        '''
        if not os.path.exists(rootOutputFile):os.makedirs(rootOutputFile)
        if not os.path.exists(rootOutputModel):os.makedirs(rootOutputModel)
        ####################################### PRE-PREPROCESSING OF DATA #####################################################################
        processing, requestOnDataBase, model = Preprocess(), RequestFromDataBase(), Model()
        dateNbDiTupleArray = requestOnDataBase.requestDiByDate()
        addingHidenDay=processing.matchingDateStartEnd(processing.dateBetweenStartEnd(dateNbDiTupleArray)[2], dict(dateNbDiTupleArray))
        print("Start the analyse from {} to {} : the last date in the potgresql database.\nTotal number of days : {} days.".format(processing.dateBetweenStartEnd(dateNbDiTupleArray)[0],
         processing.dateBetweenStartEnd(dateNbDiTupleArray)[1], len(addingHidenDay)))
        print("Number of days where interventions exist : {} days.".format(len(dateNbDiTupleArray)))
        df=pd.DataFrame(addingHidenDay[:,1], columns=['nbDi'])
        df.set_index(pd.to_datetime(addingHidenDay[:, 0]), inplace=True)
        graph = Graph(df)
        '''
        pltting nbDi by date, month, year with seaborn
        '''
        graph.graphNbDiByMonth()
        graph.graphNbDiByYear()
        graph.graphNbDiAllMonthEachYear()
        '''
        checking meteo datas
        '''
        dateMeteoTupleArray = requestOnDataBase.requestMeteoByDate(len(addingHidenDay))
        '''
        adding meteo datas to the dataframe
        '''
        df['mto_temp(celcius)'], df['mto_temp_min(celcius)'] = [i[1] for i in dateMeteoTupleArray], [i[2] for i in dateMeteoTupleArray]
        df['mto_temp_max(celcius)'], df['mto_pressure(hPa)']= [i[3] for i in dateMeteoTupleArray], [i[4] for i in dateMeteoTupleArray]
        df['mto_humidity(%)'], df['mto_visibility(km)'] = [i[5] for i in dateMeteoTupleArray], [i[6] for i in dateMeteoTupleArray]
        df['mto_wind_speed(m s)'], df['mto_clouds(%)'] = [i[7] for i in dateMeteoTupleArray], [i[8] for i in dateMeteoTupleArray]
        '''
        coefficient of linear regression
        '''
        print()
        print("-- The linear regression coefficient R is :", 
        processing.coefCorrelationLinear(np.linspace(start=1, stop=len(df), num=len(df)), df.nbDi.astype('int')))
        print()
        '''
        stationnarity of the dataset 
        '''
        isStat = processing.isStationnary(df)
        print(isStat[0])
        if(isStat[1] == False):
            print("The time series dataset is not stationnary !")
        '''
        generating future meteo files and load CSV future_meteo file 
        '''
        meteoFuture()
        '''
        loading the futureMeteoByDate.csv generated in the project
        '''
        futureMeteo = processing.loadCsvFile("generated/future/CSV/futureMeteoByDate.csv")
        '''
        decompose nbDi dataset into trend, seasonal, residual
        '''
        graph.graphSeasonalDecompose()
        '''
        checking the autocorrelation graph
        '''
        graph.graphAutocorrelationNbDi()
        '''
        partial autocorrelation graph
        '''
        graph.graphPartialAutocorrelationNbDi()
        '''
        saving in graph folder the graphs nbDi or meteo by date
        '''
        graph.graphNbDiMeteoByDate(TEST_SIZE)
        '''
        plot the no linear regression graph nbDi by date
        '''
        graph.noLinearRegressionNbDiByDate()
        '''
        plotting the linear regression graph of nbDi by date
        '''
        graph.linearRegressionNbDiByDate()
        '''
        plotting real data and z_score(outliers score) values
        '''
        graph.graphZScoreByDate()
        '''
        creating the new dijon dataframe
        '''
        dijon=df[["nbDi", 'mto_temp(celcius)' ,'mto_temp_min(celcius)', 'mto_temp_max(celcius)','mto_pressure(hPa)', 'mto_humidity(%)', 'mto_visibility(km)', 'mto_wind_speed(m s)', 
        'mto_clouds(%)']].astype('float')
        '''
        definition of the smote variable
        '''
        sm = SMOTE(random_state=20)
        '''
        detecting outliers from the dataset
        '''
        df_outliers = processing.detect_outliers_2(dijon)
        '''
        Dataframe dijon without outliers
        '''
        #dijon = processing.remove_outliers(dijon)
        '''
        normalisation of the dijon dataset with MinMaxScaler
        '''
        transformed_df = pd.DataFrame(data = processing.normaliseData(dijon.values), index=dijon.index)
        '''
        writing the dijon transformed data into a generated file
        '''
        try:
            transformed_file = open(rootOutputFile+"2- dijon_df_transformed.txt", "w+")
            transformed_file.write(str(transformed_df.shape)+'\n'+str(transformed_df.head(50)))
        except OSError as error:
            print("cannot not open or write in file", error)
        finally:
            transformed_file.close()
        '''
        initiate train and test dataset based on dijon dataset
        '''
        train_dataset = transformed_df.head(int(len(transformed_df)*(1-TEST_SIZE)))
        test_dataset = transformed_df.tail(len(transformed_df) - int(len(transformed_df)*(1-TEST_SIZE)))
        '''
        build the train and test dataset (or pre-process it)
        '''
        dijon_train, label_train = processing.build_dataset(train_dataset, LOOK_BACK)
        dijon_test, label_test = processing.build_dataset(test_dataset, LOOK_BACK)
        '''
        using SMOTE
        '''
        #dijon_train_reshape = dijon_train.reshape(dijon_train.shape[0] * dijon_train.shape[1], dijon_train.shape[2])
        #dijon_train, label_train = sm.fit_resample(dijon_train_reshape.astype(int), label_train.ravel().astype(int))
        '''
        updating global variables
        '''
        global_vars.x_shape = dijon_train.shape[1]
        global_vars.y_shape = dijon_train.shape[2]
        '''
        writing output files -- dijon transformed file
        '''
        try:
            file0,file1, file2, file3 = open(rootOutputFile+"1.1- outliers_dataframe.txt", "w+"),open(rootOutputFile+"1- init_dataframe.txt", "w+"), open(rootOutputFile+"3- train_dataset_transformed.txt", "w+"),open(rootOutputFile+"4- test_dataset_transformed.txt", "w+")
            file0.write(str(df_outliers.shape)+'\n'+str(df_outliers.head(100)))
            file1.write(str(dijon.shape)+'\n'+str(dijon.head(50)))
            file2.write(str(train_dataset.shape)+'\n'+str(train_dataset.head(50)))
            file3.write(str(test_dataset.shape)+'\n'+str(test_dataset.head(50)))
        except OSError as error:
            print("cannot not open or write in file", error)
        finally:
            for i in [file0,file1, file2, file3]:i.close()
        '''
        searching good hyperparameters for the model
        hyperparams generated there are replaced in the model
        '''
        '''hyper_params = fixHyperParamsGridSearch(model.buildModelLSTM, dijon_train, label_train)
        file_res = open(rootOutputFile+"hyper_params_grid_search", "a")
        file_res.write("\n"+str(hyper_params))
        file_res.close()'''
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
        ####################################### TRAINING #####################################################################
        '''
        building the RNN model
        '''
        model, model_name = model.buildModelLSTM()[0], model.buildModelLSTM()[1] 
        '''
        EarlyStopping to prevent the overfitting on the losses
        '''
        es= EarlyStopping(monitor='val_loss', patience=PATIENCE), 
        #history = model.fit(dijon_train, label_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
        #batch_size=BATCH_SIZE, callbacks=[es])
        history = model.fit(dijon_train, label_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
        batch_size=BATCH_SIZE)
        model.save(rootOutputModel+model_name+'.h5')
        '''
        getting the RNN model result
        '''
        evaluate = Evaluate(model)
        print(evaluate.evaluateModel(dijon_train, label_train, dijon_test, label_test))
        '''
        saving the graph of model history
        '''
        graph.graphHistoryModel(history)
        ####################################### PREDICTIONS #####################################################################
        print("\n-- Evaluation train on prediction")
        '''
        evaluation of train values predicted
        '''
        label_train = np.repeat(label_train, dijon.shape[1], axis=-1)
        y_train = (processing.unormaliseData(label_train)[:,0]).reshape(label_train.shape[0], 1)
        '''
        train values on predicted train values
        '''
        dict_train, dict_test = {}, {}
        for i in os.listdir(rootOutputModel):
            train_predict=load_model(rootOutputModel+i).predict(dijon_train, batch_size=32, verbose = 1)
            train_predict = np.repeat(train_predict, dijon.shape[1], axis=-1)
            train_predict = (processing.unormaliseData(train_predict)[:,0]).reshape(train_predict.shape[0], 1)
            dict_train[i] = train_predict
        graph.graphTruthTrainOnPredictions(y_train, dict_train)

        train_predict=model.predict(dijon_train, batch_size=32, verbose = 1)
        train_predict = np.repeat(train_predict, dijon.shape[1], axis=-1)
        train_predict = (processing.unormaliseData(train_predict)[:,0]).reshape(train_predict.shape[0], 1)
        print('nb elements in the test dataset :',len(y_train) , 'elements')
        '''
        plotting truth and nbDi prediction
        '''
        graph.graphTruthTrainOnPrediction(y_train, train_predict)
        '''
        mesuring performances of the model on known(train) values
        '''
        evaluate.generating_errors(y_train, train_predict)
        print("\n-- Evaluation test on prediction")
        '''
        evaluation of test values predicted
        '''
        label_test = np.repeat(label_test, dijon.shape[1], axis=-1)
        y_test = (processing.unormaliseData(label_test)[:,0]).reshape(label_test.shape[0], 1)
        '''
        testing predicted values
        '''
        for i in os.listdir(rootOutputModel):
            test_predict=load_model(rootOutputModel+i).predict(dijon_test, batch_size=32, verbose = 1)
            test_predict = np.repeat(test_predict, dijon.shape[1], axis=-1)
            test_predict = (processing.unormaliseData(test_predict)[:,0]).reshape(test_predict.shape[0], 1)
            dict_test[i] = test_predict
        graph.graphTruthTestOnPredictions(y_test, dict_test)

        test_predict=model.predict(dijon_test, batch_size=32, verbose = 1)
        test_predict = np.repeat(test_predict, dijon.shape[1], axis=-1)
        test_predict = (processing.unormaliseData(test_predict)[:,0]).reshape(test_predict.shape[0], 1)
        '''
        plotting truth and nbDi prediction
        '''
        graph.graphTruthTestOnPrediction(y_test, test_predict)
        '''
        mesuring performances of the model
        '''
        evaluate.generating_errors(y_test, test_predict)
        '''
        predict feature : (nb_days_predict) days
        '''
        prediction = Prediction(model, dijon, NB_DAYS_PREDICTED, LOOK_BACK)
        feature, LAST_NB_DATA = np.rint(prediction.predictNextDays( futureMeteo)), 62
        dijon_timestamps = np.array(pd.DataFrame(df.index.astype(str)).tail(LAST_NB_DATA)).flatten()
        fromDateToNumberAfter = processing.listDatesBetweenDateAndNumber(date.fromisoformat(
            dijon_timestamps[len(dijon_timestamps)-1]), NB_DAYS_PREDICTED)
        dijon_dates = np.array(fromDateToNumberAfter, dtype='datetime64[D]').astype(str)
        graph.graphPredictNextDays(LAST_NB_DATA, NB_DAYS_PREDICTED, dijon, feature, dijon_timestamps, dijon_dates)
        '''
        graph with just feature informations
        '''
        graph.graphFeatureInfos(dijon_dates, feature, NB_DAYS_PREDICTED)
        '''
        closing the postGreSQL database 
        '''
        requestOnDataBase.closeDataBaseConnect()

    if __name__ == '__main__':
        main()