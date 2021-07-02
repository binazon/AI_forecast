import os
import pandas as pd
import requests
from datetime import *

'''
openweathermap api just give us 7 forecadt day for free using
'''
key, latitude, longitude, excludes, units ="88d8939072279f6dd1283ee42e480c19", 47.316667, 5.016667, ",".join(["current","minutely","hourly", "alerts"]), "metric"
apiFutureMeteo = "https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&exclude={}&appid={}&units={}".format(latitude,
 longitude, excludes, key, units)
rootFutureJSON, rootFutureCSV = "files/future/generated/JSON", "files/future/generated/CSV"

'''
creating generating files repositories
'''
if not os.path.exists(rootFutureJSON):os.makedirs(rootFutureJSON)
if not os.path.exists(rootFutureCSV):os.makedirs(rootFutureCSV)

'''
getting from the API meteo in next 7 days
'''
def meteoFuture():
    pathJsonFuture, pathJsonCheckedFuture = rootFutureJSON+"/futureMeteoByDate.json",  rootFutureJSON+"/futureMeteoCheckedByDate.json"
    jsonFuture, jsoncheckedFuture = open(pathJsonFuture, "w+"), open(pathJsonCheckedFuture, "w+")
    pathCsvFuture = rootFutureCSV+"/futureMeteoByDate.csv"
    try:
        response = requests.get(apiFutureMeteo)
        if(response.status_code == 200):
            '''
            checking the array from second element cause first one is about current day
            '''
            result, meteoChecked = response.json().get("daily")[1:7], []
            jsonFuture.write(str({"daily" : result}).replace("'", "\""))
            #write in file relative to specified meteo infos
            for i in result:
                meteoChecked.append({
                    "mto_date" : datetime.fromtimestamp(i["dt"]).strftime("%Y-%m-%d"),
                    "mto_temp" : i.get("temp")['day'],
                    "mto_temp_min" : i.get("temp")['min'],
                    "mto_temp_max" : i.get("temp")['max'],
                    "mto_pressure" : i.get("pressure"),
                    "mto_humidity" : i.get("humidity"),
                    "mto_wind_speed" : i.get("wind_speed"),
                    "mto_couds" : i.get("clouds")
                })
            jsoncheckedFuture.write(str(meteoChecked).replace("'", "\""))
        else:
            print("error occurred when calling the openweathermap meteo API")
    except Exception as error:
        print("error occurred when calling the openweathermap meteo API", error)
    finally:
        jsonFuture.close()
        jsoncheckedFuture.close()
        #generating CSV pathCsvCheckedFuture file
        pd.read_json(pathJsonCheckedFuture).to_csv(pathCsvFuture, index=None)