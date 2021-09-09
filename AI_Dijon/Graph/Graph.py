import os
from scipy.sparse.construct import random
from scipy.stats import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import *
from scipy.stats.stats import *
from scipy.optimize import curve_fit
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime as dt
plt.style.use('seaborn')
plt.rcParams.update({'figure.max_open_warning': 0})

'''
this class used to plot graphs
'''
class Graph :
    
    '''
    constructor of the class of the connexion
    '''
    def __init__(self, df):
        self.df = df
        self.pathInput = "generated/graphs/input/samples/"
        self.pathInputAdvanced = "generated/graphs/input/advanced/"
        self.pathInputAnalysis = "generated/graphs/input_analysis/outliers/"
        self.pathOutput = "generated/graphs/output/"
        self.pathInputRegression = "generated/graphs/input/linear_regression/"
        self.pathInputAutocorrelation = "generated/graphs/input_analysis/autocorrelation/"
        self.pathInputSeasonalDecompose = "generated/graphs/input_analysis/seasonal_decompose/"
        self.color = ['red', 'black', 'green', 'violet', 'yellow', 'cyan', 'orange', 'magenta', 'pink']
        #generating folders root path
        if not os.path.exists(self.pathInput):os.makedirs(self.pathInput)
        if not os.path.exists(self.pathInputAdvanced):os.makedirs(self.pathInputAdvanced)
        if not os.path.exists(self.pathInputRegression):os.makedirs(self.pathInputRegression)
        if not os.path.exists(self.pathInputAnalysis):os.makedirs(self.pathInputAnalysis)
        if not os.path.exists(self.pathOutput):os.makedirs(self.pathOutput)
        if not os.path.exists(self.pathInputAutocorrelation):os.makedirs(self.pathInputAutocorrelation)
        if not os.path.exists(self.pathInputSeasonalDecompose):os.makedirs(self.pathInputSeasonalDecompose)

    '''
    creation of polynome to plot the non_linear regression graph
    '''
    def Pol(self, x, a, b, c, d,e,f):
        return a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5 + f

    ''''
    creation of monome to plot linear regresion graph
    '''
    def Mon(self, x , a, b):
        return a * x + b

    '''
    plotting graph of nbDi by year
    '''
    def graphNbDiByYear(self):
        try : 
            plt.figure(figsize=(24,11))
            sns.boxplot(x=[d.year for d in self.df.index], y=self.df.nbDi.astype(int), data=self.df)
            plt.title('nbDi by year',fontsize=20)
            plt.xlabel("date",fontsize=20)
            plt.plot()
            plt.savefig(self.pathInputAdvanced+'nbDi_by_year.png')
        except Exception as error:
            print("Error when plotting nbDiByYear", error)
        finally:
            plt.close()

    '''
    plotting graph of nbDi for all months
    '''
    def graphNbDiByMonth(self):
        try :
            plt.figure(figsize=(24,11))
            sns.boxplot(x=[d.month for d in self.df.index], y=self.df.nbDi.astype(int), data=self.df)
            plt.title('nbDi by month',fontsize=20)
            plt.xlabel("date",fontsize=20)
            plt.plot()
            plt.savefig(self.pathInputAdvanced+'nbDi_by_month_general.png')
        except Exception as error:
            print("Error when plotting nbDiByMonth", error)
        finally:
            plt.close()

    '''
    plotting graph of nbDi for all months each year
    '''
    def graphNbDiAllMonthEachYear(self):
        for y in set(list(self.df.index.year)):
            try:
                plt.figure(figsize=(24,11))
                sns.boxplot(x=[d.month for d in self.df[self.df.index.year == y].index], y=self.df[self.df.index.year == y].nbDi.astype(int), data=self.df)
                plt.title('nbDi by month '+str(y),fontsize=20)
                plt.xlabel("date",fontsize=20)
                plt.plot()
                plt.savefig(self.pathInputAdvanced+'nbDi_by_month_'+str(y)+'.png')
            finally:
                plt.close()

    '''
    saving in graphs/ folder the graphs nbDi or meteo by date
    '''
    def graphNbDiMeteoByDate(self, test_size):
        nb_days = 120
        nb_months = nb_days//30
        try:
            for i in range(len(self.df.columns)):
                if(i==0):
                    '''
                    nbDi by date the nb_months last month
                    '''
                    try:
                        ax = plt.subplots(figsize=(24,11))[1]
                        plt.title(str(nb_days)+" last_days_nbDi_by_date or last "+str(nb_months)+" months")
                        ax.text(3, 38, 'weekends in red on x axis', style='normal',fontsize=18, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
                        ax.plot(np.arange(nb_days), self.df[self.df.columns[i]][-1 * nb_days:].astype(int),'-o', color='blue', label=self.df.columns[i])
                        ax.set_xticks(np.arange(nb_days))
                        ax.set_xticklabels(np.array(self.df.index.values[-1 * nb_days:], dtype='datetime64[D]'), rotation=90)
                        for xtick in ax.get_xticklabels():
                            if(pd.to_datetime(xtick.get_text()).weekday()>4):xtick.set_color("red")
                        plt.xlabel("date",fontsize=14)
                        plt.ylabel(self.df.columns[i],fontsize=14)
                        plt.legend()
                        plt.savefig(self.pathInputAdvanced+self.df.columns[i]+'_by_date_last_'+str(nb_months)+'_months.png')
                    except Exception as error:
                        print("Error when plotting nbDiLast"+str(nb_months)+"Months", error)
                    finally:
                        plt.close()
                #plotting all feature of the dataframe by date
                plt.subplots(figsize=(24,11))
                plt.title(self.df.columns[i]+' by date')
                plt.plot(self.df[self.df.columns[i]][:int(len(self.df)*(1-test_size))].astype(float), color='blue', label=self.df.columns[i]+" --train")
                plt.plot(self.df[self.df.columns[i]][int(len(self.df)*(1-test_size)):].astype(float), color='gray', label=self.df.columns[i]+" --test")
                plt.xlabel("date",fontsize=14)
                plt.ylabel(self.df.columns[i],fontsize=14)
                plt.legend()
                plt.savefig(self.pathInput+'1.'+str(i)+ '- '+self.df.columns[i]+'_by_date.png')
        except Exception as error:
            print("Error when plotting "+str(self.df.columns[i])+"ByDate : \n", error)
        finally:
            plt.close()

    '''
    plotting the graph of linear regression of nbDi by date
    '''
    def noLinearRegressionNbDiByDate(self):
        xdata=np.linspace(start=1, stop=len(self.df), num=len(self.df))
        try:
            popt = curve_fit(self.Pol, xdata, self.df['nbDi'].astype('float'))[0]
            plt.subplots(figsize=(24,11))
            plt.title('linear_regression_nbDi_by_date.png')
            plt.scatter(xdata,self.df['nbDi'].astype('float'), marker='+', color='blue')
            plt.plot(xdata, self.Pol(xdata, *popt),color='black')
            plt.ylabel("nbDi",fontsize=14)
            plt.savefig(self.pathInputRegression+'no_linear_regression_nbDi_by_date.png')
        except Exception as error:
            print("Error when plotting no linear regression nbDi by date", error)
        finally:
            plt.close()

    '''
    plotting the graph of linear regression of nbDi by date
    '''
    def linearRegressionNbDiByDate(self):
        xdata=np.linspace(start=1, stop=len(self.df), num=len(self.df))
        try:
            popt = curve_fit(self.Mon, xdata, self.df['nbDi'].astype('float'))[0]
            plt.subplots(figsize=(24,11))
            plt.title('linear_regression_nbDi_by_date.png')
            plt.scatter(xdata,self.df['nbDi'].astype('float'), marker='+', color='blue')
            plt.plot(xdata, self.Mon(xdata, *popt),color='black')
            plt.ylabel("nbDi",fontsize=14)
            plt.savefig(self.pathInputRegression+'linear_regression_nbDi_by_date.png')
        except Exception as error:
            print("Error when plotting linear regression nbDi by date", error)
        finally:
            plt.close()
        
    '''
    getting the history model : train_losses, train_accuracy, val_losses, val_accuracy
    '''
    def graphHistoryModel(self, history):
        try:
            plt.subplots(figsize=(18,9))
            plt.plot(history.history['loss'], label = 'train_losses')
            plt.plot(history.history['mae'], label='train_mae')
            plt.plot(history.history['val_loss'], label='val_losses')
            plt.plot(history.history['val_mae'], label='val_mae')
            plt.xlabel("epoch",fontsize=14)
            plt.ylabel("validation and train values",fontsize=14)
            plt.legend()
            plt.savefig(self.pathOutput+'2- history_train.png')
        except Exception as error:
            print("Error when plotting history model", error)
        finally:
            plt.close()
    
    '''
    plotting truth of train values and nbDi prediction
    '''
    def graphTruthTrainOnPrediction(self, y_train, train_predict):
        try:
            ax = plt.subplots(figsize=(24,11))[1]
            plt.xlabel("pas par date",fontsize=14)
            plt.ylabel("nb demandes d'intervention",fontsize=14)
            '''
            plotting nb_elmts_to_print first : predicted on truth values
            '''
            plt.title('prediction sur valeurs réelles : '+str(len(y_train))+' elements total')
            plt.plot(y_train, color="blue", label="réel nbDi")
            plt.plot(train_predict, color="green",label="prédiction train nbDi")
            ax.set_xticks(np.arange(len(y_train)))
            ax.set_xticklabels(np.array(self.df.index.values[:len(y_train)], dtype='datetime64[D]'), rotation=90)
            plt.locator_params(axis='x', nbins=25)
            plt.legend()
            plt.savefig(self.pathOutput+'3.1- predictOnTrainValues'+'.png')
        except Exception as error:
            print("Error when plotting prediction on train values", error)
        finally:
            plt.close()

    '''
    plotting truth of test values and nbDi prediction
    '''
    def graphTruthTestOnPrediction(self, y_test, test_predict):
        try:
            ax = plt.subplots(figsize=(24,11))[1]
            plt.ylabel("nb demandes d'intervention",fontsize=14)
            plt.xlabel("pas par date",fontsize=14)
            '''
            plotting nb_elmts_to_print first : predicted on truth values
            ''' 
            plt.title('prediction sur valeurs réelles : '+str(len(y_test))+' elements total')
            plt.plot(y_test,'-o', color="blue", label="réel nbDi")
            plt.plot(test_predict,'-o', color="green",label="prédiction test nbDi")
            ax.set_xticks(np.arange(len(y_test)))
            ax.set_xticklabels(np.array(self.df.index.values[-1 * len(y_test):], dtype='datetime64[D]'), rotation=90)
            plt.legend()
            plt.savefig(self.pathOutput+'3.2- predictOnTestValues'+'.png')
        except Exception as error:
            print("Error when plotting prediction on test values", error)
        finally:
            plt.close()

    '''
    plotting truth of train values and predictions of different models
    '''
    def graphTruthTrainOnPredictions(self, y_train, list_train_predict):
        p = 0
        try:
            ax = plt.subplots(figsize=(24,11))[1]
            plt.ylabel("nb demandes d'intervention",fontsize=14)
            plt.xlabel("pas par date",fontsize=14)
            '''
            plotting nb_elmts_to_print first : predicted on truth values
            ''' 
            plt.title('prediction(train_set) sur valeurs réelles : '+str(len(y_train))+' elements total')
            plt.plot(y_train, color="blue", label="réel nbDi")
            for key, value in list_train_predict.items():
                plt.plot(value, color=self.color[p],label="prédiction train nbDi "+key[:key.index('.')])
                p+=1
            ax.set_xticks(np.arange(len(y_train)))
            ax.set_xticklabels(np.array(self.df.index.values[-1 * len(y_train):], dtype='datetime64[D]'), rotation=90)
            plt.locator_params(axis='x', nbins=25)
            plt.legend()
            plt.savefig(self.pathOutput+'3.3- predictionAllModelOnTrainValues'+'.png')
        except Exception as error:
            print("Error when plotting prediction on train values", error)
        finally:
            plt.close()

    '''
    plotting truth of test values and predictions of different models
    '''
    def graphTruthTestOnPredictions(self, y_test, list_test_predict):
        p = 0
        try:
            ax = plt.subplots(figsize=(24,11))[1]
            plt.ylabel("nb demandes d'intervention",fontsize=14)
            plt.xlabel("pas par date",fontsize=14)
            '''
            plotting nb_elmts_to_print first : predicted on truth values
            ''' 
            plt.title('prediction(test_set) sur valeurs réelles : '+str(len(y_test))+' elements total')
            plt.plot(y_test, color="blue", label="réel nbDi")
            for key, value in list_test_predict.items():
                plt.plot(value, color=self.color[p],label="prédiction test nbDi "+key[:key.index('.')])
                p+=1
            ax.set_xticks(np.arange(len(y_test)))
            ax.set_xticklabels(np.array(self.df.index.values[-1 * len(y_test):], dtype='datetime64[D]'), rotation=90)
            plt.legend()
            plt.savefig(self.pathOutput+'3.4- predictionAllModelOnTestValues'+'.png')
        except Exception as error:
            print("Error when plotting prediction on test values", error)
        finally:
            plt.close()

    '''
    predict feature : (nb_days_predict) days
    '''
    def graphPredictNextDays(self, last_data, NB_DAYS_PREDICTED, dijon, feature, dijon_timestamps, dijon_dates):
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
            plt.plot(*zip(*sorted(zip(dijon_dates, feature))), '-o', color='orange', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
            plt.legend()
            plt.savefig(self.pathOutput+'4- last '+str(last_data)+' jours + predict_'+str(NB_DAYS_PREDICTED)+'_next_days.png')
        except Exception as error:
            print("Error when plotting next predicted days", error)
        finally:
            plt.close()

    '''
    graph with just feature informations
    '''
    def graphFeatureInfos(self, dijon_dates, feature, NB_DAYS_PREDICTED):
        try:
            plt.subplots(figsize=(24,11))
            plt.xticks(rotation=90)
            plt.bar(dijon_dates, feature, color='orange', label=str(NB_DAYS_PREDICTED) + ' feature(s)')
            plt.title(str(NB_DAYS_PREDICTED) + ' next predictions values (bar chart)')
            plt.xlabel("date de demande d'intervention",fontsize=14)
            plt.ylabel("nombre de demande d'intervention",fontsize=14)
            plt.legend()
            plt.savefig(self.pathOutput+'5- predict_'+str(NB_DAYS_PREDICTED)+'_next_days.png')
        except Exception as error:
            print("Error when plotting next predicted days", error)
        finally:
            plt.close()

    '''
    this graph allows to show the z_score that we get for eatch value of nbDi dataset

    values thaht are higher than 3 are the outliers.
    '''
    def graphZScoreByDate(self):
        data_frame  = self.df.astype('float')
        z_scores = pd.DataFrame(zscore(data_frame))
        try:
            for i in range(data_frame.shape[1]):
                #plotting all feature of the dataframe by date
                plt.subplots(figsize=(24,11))
                plt.text(0, 10, 'value is outlier if z_score(value) > limit z_score', style='normal',fontsize=30, bbox={'facecolor': 'none','alpha': 0.5, 'pad': 10})
                plt.title('z_score '+data_frame.columns[i]+' by date')
                plt.plot(self.df.index, data_frame[data_frame.columns[i]].astype(float), color='blue', label=data_frame.columns[i])
                plt.plot(self.df.index,z_scores.iloc[:,i], color='black', label="z_score "+data_frame.columns[i])
                plt.axhline(y=3, linewidth=2, color='r', label='limit z_score == 3')
                plt.xlabel("date",fontsize=14)
                plt.ylabel("z_score_"+data_frame.columns[i],fontsize=14)
                plt.legend()
                plt.savefig(self.pathInputAnalysis+'1.'+str(i)+ '- z_score_'+data_frame.columns[i]+'_by_date.png')
        except Exception as error:
            print("Error when plotting z_score", error)
        finally:
            plt.close()

    '''
    plotting the seasonal decompose graph
    '''
    def graphSeasonalDecompose(self):
        try:
            decomposed = seasonal_decompose(self.df.nbDi.astype('int'), model='additive')
            trend, seasonal, residual = decomposed.trend, decomposed.seasonal, decomposed.resid
            plt.figure(figsize=(24,11))
            plt.subplot(411)
            plt.plot(self.df.nbDi.astype('int'), label = 'Original', color = 'blue')
            plt.legend(loc='upper right')
            plt.title("Original")
            plt.subplot(412)
            plt.plot(trend, label = 'Tendance', color = 'black')
            plt.legend(loc='upper right')
            plt.title("Tendance")
            plt.subplot(413)
            plt.plot(seasonal, label = 'Saisonnière', color = 'black')
            plt.legend(loc='upper right')
            plt.title("Saisonniere")
            plt.subplot(414)
            plt.plot(residual, label = 'Résidus', color = 'black')
            plt.legend(loc='upper right')
            plt.title("Residus")
            plt.savefig(self.pathInputSeasonalDecompose+'seasonal_decompose.png')
        except Exception as error:
            print("Error when plotting seasonal decompose", error)
        finally:
            plt.close()

    '''
    plotting the graph of autocorrelation on nbDi
    the autocorrelation is specifiying the number of history day (LOOKBACK) to consider for the learning : data pre_precessing

    help to check the p in the ARIMA model (p = 80)
    '''
    def graphAutocorrelationNbDi(self):
        try:
            plt.figure(figsize=(24,11))
            autocorrelation = autocorrelation_plot(self.df.nbDi.astype('int'))
            #print("best autocorrelation is", df.nbDi.astype('int').autocorr())
            autocorrelation.plot()
            plt.title('Autocorrelation on nbDi')
            plt.xlabel("Lags or number of days in the database",fontsize=14)
            plt.xlim([0,50])
            plt.plot()
            plt.savefig(self.pathInputAutocorrelation+'autocorrelation_nbDi.png')
        except Exception as error:
            print("Error when plotting autocorrelation on nbDi", error)
        finally:
            plt.close()

    '''
    plotting the partial autocorrelation graph

    help to checkthe q in the ARIMA model (q = 7)
    '''
    def graphPartialAutocorrelationNbDi(self):
        try:
            plt.figure(figsize=(24,11))
            plot_pacf(self.df.nbDi.astype('int'))
            plt.title('Partial autocorrelation on nbDi')
            plt.xlabel("Lags or number of days in the database",fontsize=14)
            plt.plot()
            plt.savefig(self.pathInputAutocorrelation+'partial_autocorrelation_nbDi.png')
        except Exception as error:
            print("Error when plotting partial autocorrelation on nbDi", error)
        finally:
            plt.close()