import pandas as pd
import pyodbc 
import os

class GetData:
    """
    A class to retrieve data from a SQL Server database and CSV files.

    Attributes:
        server (str): The server name.
        database (str): The database name.
        username (str): The username for authentication.
        password (str): The password for authentication.
        data_path (str): The path to the directory containing CSV files.

    Methods:
        db_conn(): Establishes a connection to the SQL Server database and retrieves data from a specific table.
        products(): Reads and returns product data from a CSV file.
        products_id(): Reads and returns product ID data from a CSV file.
    """

    def __init__(self, server, database, username, password, data_path = os.path.dirname(os.path.abspath(__file__))+'/'):
        """
        Initializes GetData class with connection parameters.

        Args:
            server (str): The server name.
            database (str): The database name.
            username (str): The username for authentication.
            password (str): The password for authentication.
            data_path (str): The path to the directory containing CSV files.
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.data_path = data_path

        if '\\' in self.data_path:
            self.data_path = self.data_path.replace('\\', '/')

        self.cnxn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password
        )

    def db_conn(self):
        """
        Establishes a connection to the SQL Server database and retrieves data from a specific table.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved data.
        """
        query_df_SQL_invent = '''
        SELECT 
        [SrNumber]
        ,[Quantity]
        ,[InventId]
        ,[UserNumber]
        ,[TimeStamp]
        FROM D365.I0516History 
        '''
        df_SQL_invent = pd.read_sql_query(query_df_SQL_invent, self.cnxn)

        return df_SQL_invent
    
    def products(self):
        """
        Reads product data from a CSV file and returns it as a DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing product data.
        """
        df_prod = pd.read_csv(self.data_path + 'input_data/Product.csv')[['Product', 'ProductGroup']]
        return df_prod
    
    def products_id(self):
        """
        Reads product ID data from a CSV file and returns it as a DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing product ID data.
        """
        df_prod_id = pd.read_csv(self.data_path + 'input_data/ProductIdList.csv')[['IdNumber', 'ProductGroup']]
        return df_prod_id
