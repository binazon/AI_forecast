import psycopg2

#connect to the postgresql database
def connect_to_db(username: str, passwd:str, db: str, hostServer : str, portServer : str) -> str:
    conn = None
    try:
        conn = psycopg2.connect(user = username, password = passwd, dbname = db, host=hostServer, port = portServer)
    except(Exception, psycopg2.Error) as error:
        print("Error connecting postgresql database", error)
    return conn