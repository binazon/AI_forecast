import psycopg2
import requests
from datetime import *

''''
this code is turnning on the servers (development, pre-production and production) 
and is executed each day at 00 o'clock 
'''


city, key, units, today = "Dijon,FR", "88d8939072279f6dd1283ee42e480c19", "metric", date.today().strftime("%Y-%m-%d")
apiCurrentMeteo = "https://api.openweathermap.org/data/2.5/weather?q="+str(city)+"&APPID="+str(key)+"&units="+str(units)
user_data, pass_data, db_data, host_data, port_data = "dme", "dme", "dme_ai", "frpardeml1l", "15432"

#connect to the postgresql database
def connect_to_db(username: str, passwd:str, db: str, hostServer : str, portServer : str) -> str:
    return psycopg2.connect(user = username, password = passwd, dbname = db, host=hostServer, port = portServer)

#feed the database with the current day meteo informations
try:
    insert_command = "insert into t_meteo_mto values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    conn = connect_to_db(user_data, pass_data, db_data,host_data, port_data)
    cursor = conn.cursor()
    response = requests.get(apiCurrentMeteo)
    cursor.execute("select count(*) from t_meteo_mto where mto_date='"+today+"'")
    if(response.status_code == 200 and cursor.fetchone()[0]==0):
        cursor.execute(insert_command, ("'"+today+"'", response.json().get('main')['temp'], response.json().get('main')['temp_min'],
        response.json().get('main')['temp_max'], response.json().get('main')['pressure'], response.json().get('main')['humidity'],
        response.json().get('visibility'), response.json().get('wind')['speed'],response.json().get('clouds')['all']))
        conn.commit()
        print(cursor.rowcount, "new line inserted succesfully into meteo table")
    else:
        print("This date and meteo informations already exist in the database")
except(Exception, psycopg2.Error) as error:
    print("Error fetching data from postgreSQL table", error)
finally:
    if conn is not None:
        cursor.close()
        conn.close()