import os
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
    def __init__(self):
        self.pathInput = "generated/graphs/input/samples/"
        self.pathInputAdvanced = "generated/graphs/input/advanced/"
        self.pathInputAnalysis = "generated/graphs/input_analysis/outliers/"
        self.pathOutput = "generated/graphs/output/"
        self.pathInputRegression = "generated/graphs/input/linear_regression/"
        self.pathInputAutocorrelation = "generated/graphs/input_analysis/autocorrelation/"
        self.pathInputSeasonalDecompose = "generated/graphs/input_analysis/seasonal_decompose/"
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

    '''
    plotting graph of nbDi by year
    '''
    def graphNbDiByYear(self, df):
        try : 
            plt.figure(figsize=(24,11))
            sns.boxplot(x=[d.year for d in df.index], y=df.nbDi.astype(int), data=df)
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
    def graphNbDiByMonth(self, df):
        try :
            plt.figure(figsize=(24,11))
            sns.boxplot(x=[d.month for d in df.index], y=df.nbDi.astype(int), data=df)
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
    def graphNbDiAllMonthEachYear(self, df):
        for y in set(list(df.index.year)):
            try:
                plt.figure(figsize=(24,11))
                sns.boxplot(x=[d.month for d in df[df.index.year == y].index], y=df[df.index.year == y].nbDi.astype(int), data=df)
                plt.title('nbDi by month '+str(y),fontsize=20)
                plt.xlabel("date",fontsize=20)
                plt.plot()
                plt.savefig(self.pathInputAdvanced+'nbDi_by_month_'+str(y)+'.png')
            finally:
                plt.close()

    '''
    saving in graphs/ folder the graphs nbDi or meteo by date
    '''
    def graphNbDiMeteoByDate(self, df):
        nb_days = 90
        try:
            for i in range(1, len(df.columns)):
                if(i==1):
                    '''
                    nbDi by date the 3 last month -- get more details
                    '''
                    try:
                        ax = plt.subplots(figsize=(24,11))[1]
                        plt.title(str(nb_days)+" last_days_nbDi_by_date")
                        ax.text(3, 38, 'weekends in red on x axis', style='normal',fontsize=18, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
                        ax.plot(np.arange(nb_days), df[df.columns[1]][-nb_days:].astype(float),'-o', color='blue', label=df.columns[1])
                        ax.set_xticks(np.arange(nb_days))
                        ax.set_xticklabels(np.array(df.index.values[-1 * nb_days:], dtype='datetime64[D]'), rotation=90)
                        for xtick in ax.get_xticklabels():
                            if(pd.to_datetime(xtick.get_text()).weekday()>4):xtick.set_color("red")
                        plt.xlabel("date",fontsize=14)
                        plt.ylabel(df.columns[1],fontsize=14)
                        plt.legend()
                        plt.savefig(self.pathInputAdvanced+df.columns[1]+'_by_date_last_3_months.png')
                    except Exception as error:
                        print("Error when plotting nbDiLast3Months", error)
                    finally:
                        plt.close()
                #plotting all feature of the dataframe by date
                plt.subplots(figsize=(24,11))
                plt.title(df.columns[i]+' by date')
                plt.plot(df[df.columns[i]].astype(float), color='blue', label=df.columns[i])
                plt.xlabel("date",fontsize=14)
                plt.ylabel(df.columns[i],fontsize=14)
                plt.legend()
                plt.savefig(self.pathInput+'1.'+str(i)+ '- '+df.columns[i]+'_by_date.png')
        except Exception as error:
            print("Error when plotting "+str(df.columns[i])+"ByDate : \n", error)
        finally:
            plt.close()

    '''
    plotting the graph of linear regression of nbDi by date
    '''
    def linearRegressionNbDiByDate(self, df):
        xdata=np.linspace(start=1, stop=len(df), num=len(df))
        try:
            popt, pcov = curve_fit(self.Pol, xdata, df['nbDi'].astype('float'))
            plt.subplots(figsize=(24,11))
            plt.title('linear_regression_nbDi_by_date.png')
            plt.scatter(xdata,df['nbDi'].astype('float'), marker='+', color='blue')
            plt.plot(xdata, self.Pol(xdata, *popt),color='black')
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
    plotting truth and nbDi prediction

    sign of nb_elmnts_to_print design the kind of prediction on thruth will be plotted
    positive nb_elmnts_to_print : plot nb_elmnts_to_print first elements
    negative nb_elmnts_to_print : plot nb_elmnts_to_print last elements
    '''
    def graphTruthOnPrediction(self, nb_elmnts_to_print, y_test, test_predict):
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
                plt.savefig(self.pathOutput+'3- predictOnTest_'+str(nb_elmnts_to_print)+'_first.png')
            elif(nb_elmnts_to_print < 0):
                '''
                plotting nb_elmts_to_print last : predicted on truth values
                '''
                plt.title('prediction sur valeurs réelles : ('+str(abs(nb_elmnts_to_print))+' derniers éléments) sur '+str(len(y_test))+' elements total')
                plt.plot(y_test[nb_elmnts_to_print:],'-o', color="blue", label="réel nbDi")
                plt.plot(test_predict[nb_elmnts_to_print:],'-o', color="green",label="prédiction nbDi")
                plt.legend()
                plt.savefig(self.pathOutput+'3- predictOnTest_'+str(abs(nb_elmnts_to_print))+'_last.png')
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
    def graphZScoreByDate(self, df):
        df_without_date = df.iloc[:, 1:].astype('float')
        z_scores = pd.DataFrame(zscore(df_without_date))
        try:
            for i in range(df_without_date.shape[1]):
                #plotting all feature of the dataframe by date
                plt.subplots(figsize=(24,11))
                plt.text(0, 10, 'value is outlier if z_score(value) > limit z_score', style='normal',fontsize=30, bbox={'facecolor': 'none','alpha': 0.5, 'pad': 10})
                plt.title('z_score '+df_without_date.columns[i]+' by date')
                plt.plot(df.index, df_without_date[df_without_date.columns[i]].astype(float), color='blue', label=df_without_date.columns[i])
                plt.plot(df.index,z_scores.iloc[:,i], color='black', label="z_score "+df_without_date.columns[i])
                plt.axhline(y=3, linewidth=2, color='r', label='limit z_score == 3')
                plt.xlabel("date",fontsize=14)
                plt.ylabel("z_score_"+df_without_date.columns[i],fontsize=14)
                plt.legend()
                plt.savefig(self.pathInputAnalysis+'1.'+str(i)+ '- z_score_'+df_without_date.columns[i]+'_by_date.png')
        except Exception as error:
            print("Error when plotting z_score", error)
        finally:
            plt.close()

    '''
    plotting the seasonal decompose graph
    '''
    def graphSeasonalDecompose(self, df):
        try:
            decomposed = seasonal_decompose(df.nbDi.astype('int'), model='additive')
            trend, seasonal, residual = decomposed.trend, decomposed.seasonal, decomposed.resid
            plt.figure(figsize=(24,11))
            plt.subplot(411)
            plt.plot(df.nbDi.astype('int'), label = 'Original', color = 'blue')
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
    def graphAutocorrelationNbDi(self, df):
        try:
            plt.figure(figsize=(24,11))
            autocorrelation = autocorrelation_plot(df.nbDi.astype('int'))
            #print(df.nbDi.astype('int').autocorr())
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
    def graphPartialAutocorrelationNbDi(self, df):
        try:
            plt.figure(figsize=(24,11))
            plot_pacf(df.nbDi.astype('int'))
            plt.title('Partial autocorrelation on nbDi')
            plt.xlabel("Lags or number of days in the database",fontsize=14)
            plt.plot()
            plt.savefig(self.pathInputAutocorrelation+'partial_autocorrelation_nbDi.png')
        except Exception as error:
            print("Error when plotting partial autocorrelation on nbDi", error)
        finally:
            plt.close()