from datetime import date, datetime, timedelta
from pathlib import Path
import os
from typing import List
import numpy as np
import pandas as pd
import json
import requests
import dateutil.parser

zipDijon, countryDijon, rootJson, rootCsv, city  = 21000, 'FR', "files/history/generated/JSON", "files/history/generated/CSV", "Dijon,FR"
keys = ["b876fe5cfec442e4b54c42b8a570b43b","a7a78baab96c4db19e786a7470fe1559", "8e160363f6aa45449161b131395189b1","e3c45801fb864127b62682285a7d337f"]
apiHistory = "https://api.weatherbit.io/v2.0/history/daily?postal_code="+str(zipDijon)+"&country="+str(countryDijon)+"&start_date={}&end_date={}&key={}"
apiFuture = "https://api.weatherbit.io/v2.0/forecast/daily?city="+str(city)+"&key={}"
if not os.path.exists(rootJson):os.makedirs(rootJson)
if not os.path.exists(rootCsv):os.makedirs(rootCsv)

#reading content of a file
def readFileContent(path) -> str:
    content =''
    with open(path, 'r') as file:
        content = file.read()
    file.close()
    return content

#generating the history json & csv history file
def generateHistoryMeteo(_array) -> int:
    pathJsonHistory = rootJson+"/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    pathJsonHistoryCheck = rootJson+"/historyMeteoChecked"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    historyMeteoJSON, historyMeteoJSONChecked = open(pathJsonHistory,'a+'), open(pathJsonHistoryCheck,'a+')
    pathCsvGenerated = rootCsv+"/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".csv"
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
        element,i,array, checkedArray=0,0,[],[]
        while(i < len(alldays)-1):
            try:
                response = requests.get(apiHistory.format(datetime.date(dateutil.parser.parse(str(alldays[i]))),datetime.date(dateutil.parser.parse(str(alldays[i+1]))),keys[element]))
                if(response.status_code == 429):
                    print("error occured when checking api.weatherbit.io on the date : "+str(datetime.date(dateutil.parser.parse(str(alldays[i]))))+" for key : " +str(keys[element]))
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
    response = requests.get(apiFuture.format(keys[0]))
    lastFutureMeteo = date.fromisoformat(_array[len(_array)-1][1])+timedelta(days=nbDaysPredict)
    pathJsonFuture = rootJson+"/futureMeteo"+str(_array[len(_array)-1][1])+"_"+str(lastFutureMeteo)+".json"
    if(_array[len(_array)-1][1] < str(date.today())):
        #get difference between current date and last date in database
        diff = abs(date.fromisoformat(str(date.today())) - date.fromisoformat(_array[len(_array)-1][1])).days
        print(diff)
        if(diff < nbDaysPredict):
            #priting the history
            alldays = np.array(pd.date_range(date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=1), date.fromisoformat(str(date.today()))))
            #pritting the prevision
            alldaysFuture = np.array(pd.date_range(date.fromisoformat(str(date.today()))+timedelta(days=1), date.fromisoformat(str(date.today())) + timedelta(days=nbDaysPredict - diff)))
        else:
            #priting the history
            alldays = np.array(pd.date_range(date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=1), date.fromisoformat(str(_array[len(_array)-1][1]))+timedelta(days=nbDaysPredict)))
    elif(_array[len(_array)-1][1] == str(date.today())):
        #priting the prevision
        alldaysFuture = np.array(pd.date_range(date.fromisoformat(str(date.today()))+timedelta(days=1), date.fromisoformat(str(date.today())) + timedelta(days=nbDaysPredict)))
    return -1