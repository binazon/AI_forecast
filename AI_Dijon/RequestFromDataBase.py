import os,sys
#add parent folder path to list of sys.path
sys.path.insert(1, os.path.abspath('.'))
from datetime import *
from Preprocessing import *
from connect_database.ConnectPostGreSQL import *

class RequestFromDataBase :
   
    '''
    defining the constructor of the class
    '''
    def __init__(self) -> None:
        self.conn = ConnectPostGreSQL().connect_to_db()

    '''
    get lines or datas of the view di_by_date
    return array of tuples (date, number intervention)
    '''
    def requestDiByDate(self) -> List:
        try:
            cursor = self.conn.cursor()
            '''
            getting nbDi of each date
            '''
            cursor.execute("select * from di_by_date")
            diByDateInBddArray = cursor.fetchall()
        except Exception as error:
            print('Error fetching data from postgreSQL table', error)
        finally:
            if self.conn is not None:
                cursor.close()
        return diByDateInBddArray

    '''
    get lines or datas of the table t_meteo_mto
    return array of tuple (date, meteo datas) 
    '''
    def requestMeteoByDate(self) -> List:
        try:
            cursor = self.conn.cursor()
            '''
            getting meteo of each date
            '''
            cursor.execute("select * from t_meteo_mto order by mto_date")
            meteoByDateInBddArray = cursor.fetchall()
            for i in range(len(meteoByDateInBddArray)):
                meteoTupleToList = list(meteoByDateInBddArray[i])
                meteoTupleToList[0] = meteoTupleToList[0].strftime("%Y-%m-%d")
                meteoByDateInBddArray[i]=tuple(meteoTupleToList)
        except Exception as error:
            print('Error fetching data from postgreSQL table', error)
        finally:
            if self.conn is not None:
                cursor.close()
        return meteoByDateInBddArray

    '''
    get lines or datas of the table t_meteo_mto

    return array of tuple (date, meteo datas) 
    '''
    def requestMeteoByDate(self, _limit : int) -> List:
        try:
            cursor = self.conn.cursor()
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
            if self.conn is not None:
                cursor.close()
        return meteoByDateInBddArray

    '''
    close the connection of the postgreSQL database
    '''
    def closeDataBaseConnect(self):
        self.conn.close()
        print("Connextion to the postgreSQL database closed !")