from datetime import date, datetime, timedelta
from pathlib import Path
import os
from typing import List
import numpy as np
from numpy.core.records import array
import pandas as pd
import json
import requests
import dateutil.parser

zipDijon, countryDijon, rootHistoryJson, rootHistoryCsv, city  = 21000, 'FR', "files/history/generated/JSON", "files/history/generated/CSV", "Dijon,FR"
rootFutureJson, rootFutureCsv = "files/future/generated/JSON", "files/future/generated/CSV"
keys = ["debb42dcc5874a48a7b69b7638d50639","a7a78baab96c4db19e786a7470fe1559", "8e160363f6aa45449161b131395189b1","e3c45801fb864127b62682285a7d337f"]
apiHistory = "https://api.weatherbit.io/v2.0/history/daily?postal_code="+str(zipDijon)+"&country="+str(countryDijon)+"&start_date={}&end_date={}&key={}"
apiFuture = "https://api.weatherbit.io/v2.0/forecast/daily?city="+str(city)+"&key={}"
if not os.path.exists(rootHistoryJson):os.makedirs(rootHistoryJson)
if not os.path.exists(rootHistoryCsv):os.makedirs(rootHistoryCsv)
if not os.path.exists(rootFutureJson):os.makedirs(rootFutureJson)
if not os.path.exists(rootFutureCsv):os.makedirs(rootFutureCsv)

#from a datetime string to date 
def datetime_to_date(enter) -> str:
    return datetime.date(dateutil.parser.parse(str(enter)))

#reading content of a file
def readFileContent(path) -> str:
    content =''
    with open(path, 'r') as file:
        content = file.read()
    file.close()
    return content

#getting the history of meteo from api
def historyMeteoDictObject(alldays) -> str:
    element,i,array, checkedArray=0,0,[],[]
    while(i < len(alldays)-1):
        try:
            response = requests.get(apiHistory.format(datetime_to_date(alldays[i]),datetime_to_date(alldays[i+1]),keys[element]))
            if(response.status_code == 429):
                print("error occured when checking api.weatherbit.io on the date : "+str(datetime_to_date(alldays[i]))+" for key : " +str(keys[element]))
                if(element==len(keys)-1):
                    print("not anough keys to request the api")
                    break
                else:
                    element+=1
                    i-=1
            elif(response.status_code == 200):
                array.append(str(response.json()).replace("'", "\"").replace("None","\"None\""))
                checkedArray.append(str(response.json()['data'][0]).replace("'", "\"").replace("None","\"None\""))
            else:
                print("cannot acces the api api.weatherbit.io")
            i+=1
        except KeyboardInterrupt:
            # nee to quit
            print("En cours... votre programme génère les fichiers d'historique météo. Veuillez attendre...")
    return array, checkedArray

#generating the history json & csv history file
def generateHistoryMeteo(_array) -> int:
    pathJsonHistory = rootHistoryJson+"/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    pathJsonHistoryCheck = rootHistoryJson+"/historyMeteoChecked"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    historyMeteoJSON, historyMeteoJSONChecked = open(pathJsonHistory,'a+'), open(pathJsonHistoryCheck,'a+')
    pathCsvGenerated = rootHistoryCsv+"/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".csv"
    contentJsonHistoryFile, contentJsonHistoryChecked = "", ""
    #reading last date of the Json hitory file
    frame = open(pathJsonHistory,)
    if(os.stat(pathJsonHistory).st_size > 0):
        data = json.load(frame)
        frame.close()
        lastDateHistoryFile = data[len(data)-1]['data'][0]['datetime']
        #reading the content of the Json history file
        contentJsonHistoryFile = readFileContent(pathJsonHistory)
        contentJsonHistoryChecked = readFileContent(pathJsonHistoryCheck)
        alldays = np.array(pd.date_range(date.fromisoformat(lastDateHistoryFile)+timedelta(days=1), date.fromisoformat(_array[len(_array)-1][1])+timedelta(days=1)))
    else:
        #case history Json file is empty
        lastDateHistoryFile = _array[0][1]
        alldays = np.array(pd.date_range(date.fromisoformat(lastDateHistoryFile), date.fromisoformat(_array[len(_array)-1][1])+timedelta(days=1)))
    #running the code if we nee to add some history
    if(lastDateHistoryFile < _array[len(_array)-1][1]):
        array, checkedArray = historyMeteoDictObject(alldays)
        historyMeteoJSON.truncate(0)
        historyMeteoJSON.write(str('[' + ("" if contentJsonHistoryFile=="" else str(contentJsonHistoryFile.strip("[]"))+',') + ',\n'.join(array)+']'))
        historyMeteoJSONChecked.truncate(0)
        historyMeteoJSONChecked.write(str('['+  ("" if contentJsonHistoryChecked=="" else str(contentJsonHistoryChecked.strip("[]"))+',') +',\n'.join(checkedArray)+']'))
        historyMeteoJSON.close()
        historyMeteoJSONChecked.close()
        dataFrame = pd.read_json(pathJsonHistoryCheck)
        dataFrame.to_csv(pathCsvGenerated, index=None)
    return -1

#generate future meteo datas
def generateFutureMeteo(_array, nbDaysPredict) -> int:
    pathJsonFuture = rootFutureJson+"/futureMeteoFrom{}To{}.json"
    checkedArray = []
    response = requests.get(apiFuture.format(keys[0]))
    if(_array[len(_array)-1][1] < str(date.today())):
        #get difference between current date and last date in database
        diff = abs(date.fromisoformat(str(date.today())) - date.fromisoformat(_array[len(_array)-1][1])).days
        if(diff < nbDaysPredict):
            #get the history
            allHistorydays = np.array(pd.date_range(date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=1), date.fromisoformat(str(date.today()))))
            checkedArray = historyMeteoDictObject(allHistorydays)[1]
            #get the future
            alldaysFuture = np.array(pd.date_range(date.fromisoformat(str(date.today()))+timedelta(days=1), date.fromisoformat(str(date.today())) + timedelta(days=nbDaysPredict - diff)))
            pathJsonFuture = pathJsonFuture.format(allHistorydays[0], alldaysFuture[len(alldaysFuture)])
            for i in alldaysFuture:
                print(i)
        else:
            #get the history
            allHistorydays = np.array(pd.date_range(date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=1), date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=nbDaysPredict)))
            checkedArray = historyMeteoDictObject(allHistorydays)[1]
            pathJsonFuture = pathJsonFuture.format(datetime_to_date(allHistorydays[0]), datetime_to_date(allHistorydays[len(allHistorydays)-1]))
            
    elif(_array[len(_array)-1][1] == str(date.today())):
        #get the future
        alldaysFuture = np.array(pd.date_range(date.fromisoformat(str(date.today()))+timedelta(days=1), date.fromisoformat(str(date.today())) + timedelta(days=nbDaysPredict)))
        pathJsonFuture = pathJsonFuture.format(alldaysFuture[0], alldaysFuture[len(alldaysFuture)])
        for i in alldaysFuture:
            print(i)
    futureMeteoJSON = open(pathJsonFuture,'a+')
    futureMeteoJSON.truncate(0)
    contentJsonFuture = readFileContent(pathJsonFuture)
    futureMeteoJSON.write(str('['+  ("" if contentJsonFuture=="" else str(contentJsonFuture.strip("[]"))+',') +',\n'.join(checkedArray)+']'))
    futureMeteoJSON.close()
    return -1