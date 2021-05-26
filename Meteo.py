from datetime import datetime
import numpy as np
import pandas as pd
import requests
import dateutil.parser

zipDijon = 21000
countryDijon = 'FR'
keys = ["bdee2472caf54240b7a2c35cd8d607bf","7110135f82c54eb295024d882a7a4449", "03dc684caa274682a5b7a6f603a0ce3c","e3c45801fb864127b62682285a7d337f"]
api = "https://api.weatherbit.io/v2.0/history/daily?postal_code="+str(zipDijon)+"&country="+str(countryDijon)+"&start_date={}&end_date={}&key={}"

def generateJsonFile(_array) -> int:
    historyMeteoJSON = open("files/0- historyMeteo"+str(_array[0][1])+"_"+str(_array[len(_array)-1][1])+"Test.json", "w")
    alldays = np.array(pd.date_range(_array[0][1],_array[len(_array)-1][1]))
    p=0
    for i in range(len(alldays)):
        response = requests.get(api.format(datetime.date(dateutil.parser.parse(str(alldays[i]))),datetime.date(dateutil.parser.parse(str(alldays[i+1]))),keys[p]))
        if(response.status_code == 429):
            print("error occured when checking api.weatherbit.io on the date : "+str(datetime.date(dateutil.parser.parse(str(alldays[i]))))+" for key : " +str(keys[p]))
            if(p==len(keys)-1):
                print("not anough keys to request the api")
                break
            else:
                p+=1
                i-=2
        elif(response.status_code == 200):
            historyMeteoJSON.write(str(response.json())+"\n")
    return 1