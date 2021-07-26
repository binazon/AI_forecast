from typing import Dict, List
import pandas as pd
import numpy as np
import unidecode, re
from deprecated import deprecated
from sklearn.preprocessing import *
from datetime import *

'''
scale is defined from -1 to -1 to normalise datas
'''
scaler = MinMaxScaler()
#######################################METHODS#####################################################################
'''
loading csv file
'''
def loadCsvFile(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf8')

#sorting the data by date
@deprecated
def sortDijonExtractByDate(data) -> List:
    return np.array(data.sort_values('date'))

#from datetime to date
@deprecated
def datetime_array_to_date(datetimeArray) -> List:
    for i in datetimeArray:
        i[1]=i[1].split(" ")[0]
    return datetimeArray

#salting comments : removing spacial characters(without spaces) and accents and one hot comments
@deprecated
def saltingComments(array_csv) -> List:
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

#mapping by nbDi and type/comments
@deprecated
def groupByDateAndComments(array_csv) -> Dict:
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
'''
@deprecated
def groupingByDateAndDI(array) -> Dict:
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
def matchingDateStartEnd(unionDijon, nbDiByDateJson) -> List:
    dijonBddJson={}
    for union in unionDijon:
        dijonBddJson[union] = nbDiByDateJson.get(union) if(union in nbDiByDateJson.keys()) else 0
    return np.array(list(dijonBddJson.items()))

'''
date bettween start en end of all DI - even date not in bdd.
'''
def dateBetweenStartEnd(_array):
    if(type(_array[0]) is tuple):
        #from array of tuple to array of str
        _array = [i[0] for i in _array]
    return _array[0],_array[len(_array)-1], np.array(pd.date_range(_array[0],_array[len(_array)-1]), dtype='datetime64[D]').astype(str) 

'''
get list of date between date and nb following days
'''
def listDatesBetweenDateAndNumber(date, number) -> List:
    start = date + timedelta(days=1)
    end = date + timedelta(days=number)
    return np.array(pd.date_range(start, end))

'''
normalising data with values between 0 and 1
'''
def normaliseData(dataArray) -> List:
    return scaler.fit_transform(dataArray)

'''
unormalise data : from values between 0 and 1 to real values
'''
def unormaliseData(dataArray) -> List:
    return scaler.inverse_transform(dataArray)