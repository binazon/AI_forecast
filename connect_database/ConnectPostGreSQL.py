import psycopg2

#connect to the postgresql database
def connect_to_db(username: str, passwd:str, db: str, hostServer : str, portServer : str) -> str:
    return psycopg2.connect(user = username, password = passwd, dbname = db, host=hostServer, port = portServer)