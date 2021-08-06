import os,sys
#add parent folder path to list of sys.path
sys.path.insert(1, os.path.abspath('.'))
import requests
from datetime import *
from connect_database.ConnectPostGreSQL import *

''''
this code is turnning on the servers (development, pre-production and production) 
and is executed each day at 00 o'clock 
'''

city, key, units, today = "Dijon,FR", "88d8939072279f6dd1283ee42e480c19", "metric", date.today().strftime("%Y-%m-%d")
apiCurrentMeteo, conn = "https://api.openweathermap.org/data/2.5/weather?q="+str(city)+"&APPID="+str(key)+"&units="+str(units), None

#feed the database with the current day meteo informations
try:
    insert_command = "insert into t_meteo_mto values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cursor = ConnectPostGreSQL().connect_to_db().cursor()
    response = requests.get(apiCurrentMeteo)
    cursor.execute("select count(*) from t_meteo_mto where mto_date='"+today+"'")
    if(response.status_code == 200 and cursor.fetchone()[0]==0):
        cursor.execute(insert_command, ("'"+today+"'", response.json().get('main')['temp'], response.json().get('main')['temp_min'],
        response.json().get('main')['temp_max'], response.json().get('main')['pressure'], response.json().get('main')['humidity'],
        response.json().get('visibility')/1000, response.json().get('wind')['speed'],response.json().get('clouds')['all']))
        conn.commit()
        print(cursor.rowcount, "new line inserted succesfully into meteo table")
    else:
        print("This date and meteo informations already exist in the database")
except Exception as error:
    print("Error fetching data from postgreSQL table", error)
finally:
    if conn is not None:
        cursor.close()
        conn.close()