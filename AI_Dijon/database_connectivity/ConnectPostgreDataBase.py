import psycopg2

class ConnectPostgreDataBase :

    #constructor of the class of the connexion
    def __init__(self):
        self.user_data = "dme"
        self.pass_data = "dme"
        self.db_data = "dme_ai"
        self.host_data = "localhost"
        self.port_data = "5432"

    #connect to the postgresql database
    def connect_to_db(self) -> str:
        conn = None
        try:
            conn = psycopg2.connect(user = self.user_data, password = self.pass_data, dbname = self.db_data, 
            host=self.host_data, port = self.port_data)
            print("Connexion to the postGreSQL database done !")
        except(Exception, psycopg2.Error) as error:
            print("Error connecting postgresql database", error)
        return conn