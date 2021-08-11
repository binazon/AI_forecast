from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import unidecode, re
from deprecated import deprecated
from sklearn.preprocessing import *
from datetime import *
from statsmodels.tsa.stattools import *
from scipy.stats import zscore


'''
class to process of the data
'''
class Preprocess:
    def __init__(self) -> None:
        self.scaler = MinMaxScaler()

    '''
    loading csv file
    '''
    def loadCsvFile(self, path) -> pd.DataFrame:
        return pd.read_csv(path, encoding='utf8')

    '''
    sorting the data by date
    function used when work with a CSV file
    '''
    @deprecated
    def sortDijonExtractByDate(self, data) -> List:
        return np.array(data.sort_values('date'))

    '''
    from datetime to date
    function used when work with a CSV file
    '''
    @deprecated
    def datetime_array_to_date(self, datetimeArray) -> List:
        for i in datetimeArray:
            i[1]=i[1].split(" ")[0]
        return datetimeArray

    '''
    salting comments : removing spacial characters(without spaces) and accents and one hot comments
    '''
    @deprecated
    def saltingComments(self, array_csv) -> List:
        array_csv_copy=array_csv.copy()
        for i in range(0,len(array_csv_copy[:,2])):
            saltingComments=""
            for j in range(0,len(array_csv_copy[:,2][i])):
                if(str(array_csv_copy[:,2][i][j]).isalnum() or str(array_csv_copy[:,2][i][j])==' '):
                        saltingComments+=array_csv_copy[:,2][i][j]
                if(j==len(array_csv_copy[:,2][i])-1):
                    saltingComments = re.sub("[ ]{2,}", ' ',saltingComments)
                    array_csv_copy[:,2][i]=unidecode.unidecode(saltingComments).strip()
        #one hot on coments/type column by fit_trasform
        label_encoder = LabelEncoder()
        array_csv_copy[:,2] = label_encoder.fit_transform(array_csv_copy[:,2])    
        return array_csv_copy

    '''
    mapping by nbDi and type/comments
    '''
    @deprecated
    def groupByDateAndComments(self, array_csv) -> Dict:
        dict_out = {}
        nbDi = 1
        for i in range(0,len(array_csv)):
            if(array_csv[i][1] in dict_out):
                for key,values in dict_out.items():
                    if(key == array_csv[i][1]):
                        if(array_csv[i][2] in values.keys()):
                            values[array_csv[i][2]] = values.get(array_csv[i][2])+1
                        else:
                            values[array_csv[i][2]] = nbDi
            else:
                dict_out[array_csv[i][1]]={array_csv[i][2]:nbDi}    
        #print(json.dumps(dict_out, indent=4, sort_keys=True))
        return dict_out

    '''
    grouping by date and count nb DI
    function used when work with a CSV file
    '''
    @deprecated
    def groupingByDateAndDI(self, array) -> Dict:
        nbDiByDateDict={}
        nbDi=1
        for i in range(0,len(array)):
            if(array[i][1] in nbDiByDateDict):
                nbDiByDateDict[array[i][1]] = nbDiByDateDict.get(array[i][1])+1
            else:
                nbDiByDateDict[array[i][1]] = nbDi
        return nbDiByDateDict

    '''
    matching dijon date with other date in start and end
    '''
    def matchingDateStartEnd(self, unionDijon, nbDiByDateJson) -> List:
        dijonBddJson={}
        for union in unionDijon:
            dijonBddJson[union] = nbDiByDateJson.get(union) if(union in nbDiByDateJson.keys()) else 0
        return np.array(list(dijonBddJson.items()))

    '''
    date bettween start en end of all DI - even date not in bdd.
    '''
    def dateBetweenStartEnd(self, _array):
        if(type(_array[0]) is tuple):
            #from array of tuple to array of str
            _array = [i[0] for i in _array]
        return _array[0],_array[len(_array)-1], np.array(pd.date_range(_array[0],_array[len(_array)-1]), dtype='datetime64[D]').astype(str) 

    '''
    get list of date between date and nb following days
    '''
    def listDatesBetweenDateAndNumber(self, date, number) -> List:
        start = date + timedelta(days=1)
        end = date + timedelta(days=number)
        return np.array(pd.date_range(start, end))

    '''
    this method test if data is stationnary
    Argumented Dickey Fuller ADF test
    '''
    def isStationnary(self, df):
        array = adfuller(df.nbDi.astype('int'))
        res = "--- Test stationnarity : Argumented Dickey Fuller ADF test\n"
        res +=  "ADF statistic : "+str(array[0])+"\np-value : "+str(array[1])
        return (res, True) if array[0] < array[4]["5%"] else (res, False)

    '''
    this method put the data stationnary
    '''
    def stationnaryData(self, df, timestamp):
        df['nbDi_stationnary']=df.nbDi.astype('int') - df.nbDi.astype('int').shift(timestamp)

    #creating the lstm dataset
    def build_dataset(self, transformed_df, look):
        try:
            number_of_rows, number_of_features = transformed_df.shape[0], transformed_df.shape[1]
            train = np.empty([number_of_rows - look, look, number_of_features], dtype='float')
            label = np.empty([number_of_rows - look, 1])
            for i in range(number_of_rows-look):
                train[i] = transformed_df.iloc[i:i+look, 0:number_of_features]
                label[i] = transformed_df.iloc[i+look:i+look+1, 0:1]
        except Exception as error:
            print("Error in the dataset build :", error)
        return train, label 

    '''
    check another processing of the data
    '''
    def build_dataset_2(self, transformed_df, look):
        train, label = [], []
        for i in range(look, len(transformed_df)):
            train.append(transformed_df.iloc[i - look:i, 0:transformed_df.shape[1]])
            label.append(transformed_df.iloc[i:i+1,0])
        return np.array(train), np.array(label)

    '''
    this method allows to  detect outliers in our dataset for each element of the data we substracted 
    the mean and  we divide the result by the standard deviation

    return the dataframe
    '''
    def detect_outliers(self, data):
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
    def detect_outliers_2(self, df):
        z_scores = zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        return df[~filtered_entries]

    '''
    we build up the dataframe without outliers and we return a filtered dataframe
    '''
    def remove_outliers(self, df):
        z_scores = zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        return df[filtered_entries]

    '''
    normalising data with values between 0 and 1
    '''
    def normaliseData(self, dataArray) -> List:
        return self.scaler.fit_transform(dataArray)

    '''
    unormalise data : from values between 0 and 1 to real values
    '''
    def unormaliseData(self, dataArray) -> List:
        return self.scaler.inverse_transform(dataArray)