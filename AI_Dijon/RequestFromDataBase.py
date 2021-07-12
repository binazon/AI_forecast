import os,sys
#add parent folder path to list of sys.path
sys.path.insert(1, os.path.abspath('.'))
from datetime import *
from Preprocessing import *
from connect_database.ConnectPostGreSQL import *

#global variables
user_data, pass_data, db_data, host_data, port_data = "dme", "dme", "dme_ai", "localhost", "5432"

'''
get lines or datas of the view di_by_date

return array of tuples (date, number intervention)
'''
def requestDiByDate() -> List:
    try:
        conn = connect_to_db(user_data, pass_data, db_data,host_data, port_data)
        cursor = conn.cursor()
        '''
        getting nbDi of each date
        '''
        cursor.execute("select * from di_by_date")
        diByDateInBddArray = cursor.fetchall()
    except Exception as error:
        print('Error fetching data from postgreSQL table', error)
    finally:
        if conn is not None:
            cursor.close()
            conn.close()
    return diByDateInBddArray

'''
get lines or datas of the table t_meteo_mto

return array of tuple (date, meteo datas) 
'''
def requestMeteoByDate(_limit : int) -> List:
    try:
        conn = connect_to_db(user_data, pass_data, db_data,host_data, port_data)
        cursor = conn.cursor()
        '''
        getting meteo of each date
        stopping at the last date in di_by_date view
        '''
        cursor.execute("select * from t_meteo_mto order by mto_date limit {}".format(_limit))
        meteoByDateInBddArray = cursor.fetchall()
        for i in range(len(meteoByDateInBddArray)):
            meteoTupleToList = list(meteoByDateInBddArray[i])
            meteoTupleToList[0] = meteoTupleToList[0].strftime("%Y-%m-%d")
            meteoByDateInBddArray[i]=tuple(meteoTupleToList)
    except Exception as error:
        print('Error fetching data from postgreSQL table', error)
    finally:
        if conn is not None:
            cursor.close()
            conn.close()
    return meteoByDateInBddArray