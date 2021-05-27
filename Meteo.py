from datetime import date, datetime, timedelta
import os
import numpy as np
import pandas as pd
import json
import requests
import dateutil.parser
import sys

zipDijon, countryDijon  = 21000, 'FR'
keys = ["e303e0036b884f27840e765cb55010d8","bdee2472caf54240b7a2c35cd8d607bf", "03dc684caa274682a5b7a6f603a0ce3c","e3c45801fb864127b62682285a7d337f"]
api = "https://api.weatherbit.io/v2.0/history/daily?postal_code="+str(zipDijon)+"&country="+str(countryDijon)+"&start_date={}&end_date={}&key={}"

#generating the history json & csv history file
def generateHistoryMeteo(_array) -> int:
    pathJsonHistory = "files/history/generated/JSON/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    pathJsonHistoryCheck = "files/history/generated/JSON/historyMeteoChecked"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".json"
    historyMeteoJSON, historyMeteoJSONChecked = open(pathJsonHistory, "a+"), open(pathJsonHistoryCheck, "a+")
    pathCsvGenerated = "files/history/generated/CSV/historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+".csv"

    #reading last date of the Json hitory file
    frame = open(pathJsonHistory,)
    if(os.stat(pathJsonHistory).st_size > 0):
        data = json.load(frame)
        frame.close()
        lastDateHistoryFile = data[len(data)-1]['data'][0]['datetime']
    else:
        #case history Json file is emplty
        lastDateHistoryFile = _array[0][1]
    #running the code if we nee to add some history
    if(lastDateHistoryFile < _array[len(_array)-1][1]):
        element,i=0,0
        array, checkedArray = [], []
        alldays = np.array(pd.date_range(date.fromisoformat(lastDateHistoryFile) + timedelta(days=1), date.fromisoformat(_array[len(_array)-1][1])+timedelta(days=1)))
        try:
            while(i < len(alldays)-1):
                response = requests.get(api.format(datetime.date(dateutil.parser.parse(str(alldays[i]))),datetime.date(dateutil.parser.parse(str(alldays[i+1]))),keys[element]))
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
            historyMeteoJSON.write(str('['+',\n'.join(array)+']'))
            historyMeteoJSONChecked.write(str('['+',\n'.join(checkedArray)+']'))
            historyMeteoJSON.close()
            historyMeteoJSONChecked.close()
            dataFrame = pd.read_json(pathJsonHistoryCheck)
            dataFrame.to_csv(pathCsvGenerated, index=None)
        except KeyboardInterrupt:
            # nee to quit
            historyMeteoJSON.write(str('['+',\n'.join(array)+']'))
            historyMeteoJSONChecked.write(str('['+',\n'.join(checkedArray)+']'))
            historyMeteoJSON.close()
            historyMeteoJSONChecked.close()
            dataFrame = pd.read_json(pathJsonHistoryCheck)
            dataFrame.to_csv(pathCsvGenerated, index=None)
            sys.exit()

    return -1