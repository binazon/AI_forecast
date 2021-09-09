import psycopg2

'''
Singleton class to the database connexion
'''
class ConnectPostgreDataBase :
    __instance = None

    '''
    constructor of the class of the connexion
    '''
    def __init__(self):
        self.user_data = "dme"
        self.pass_data = "dme"
        self.db_data = "dme_ai"
        self.host_data = "localhost"
        self.port_data = "5432"
        if(ConnectPostgreDataBase.__instance != None):
            raise Exception("This class is a singleton")
        else:
            try:
                ConnectPostgreDataBase.__instance = psycopg2.connect(user = self.user_data, password = self.pass_data, dbname = self.db_data, 
                host=self.host_data, port = self.port_data)
                print("Connexion to the postGreSQL database done !")
            except(Exception, psycopg2.Error) as error:
                print("Error connecting postgresql database", error)

    '''
    this is the method to get the instance of connexion
    '''
    @staticmethod
    def connect_to_db():
        if ConnectPostgreDataBase.__instance == None:
            ConnectPostgreDataBase()
        return ConnectPostgreDataBase.__instance
