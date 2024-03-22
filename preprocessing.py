import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class Preprocessing:
    """
    A class for data preprocessing tasks.

    Attributes:
        df (pandas.DataFrame): The input DataFrame containing the raw data.
        products_id (pandas.DataFrame): DataFrame containing product IDs.
        products (pandas.DataFrame): DataFrame containing product information.

    Methods:
        transform_data(): Transform the raw data.
        fill_missing_value(): Fill missing values in the transformed data.
        outlier_correction_ml(): Perform outlier correction using machine learning techniques.
    """

    def __init__(self, df, products_id, products):
        """
        Initializes Preprocessing class with input data and product information.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the raw data.
            products_id (pandas.DataFrame): DataFrame containing product IDs.
            products (pandas.DataFrame): DataFrame containing product information.
        """
        self.df = df
        self.products_id = products_id
        self.products = products
 
    def transform_data(self):
        """
        Transform the raw data by filtering based on date range and merging with product IDs.

        Returns:
            pandas.DataFrame: Transformed DataFrame.
        """
        self.df = self.df[(self.df['TimeStamp'] >= '2022-05-25') & (self.df['TimeStamp'] <= '2023-11-01')]
        self.df = pd.merge(self.df, self.products_id, left_on='SrNumber', right_on='IdNumber', how='left')
        return self.df
    
    def fill_missing_value(self, window_size=2, standard_deviation=3, product_column='ProductGroup'):
        """
        Fill missing values in the transformed data using moving average and linear interpolation.

        Args:
            window_size (int): Size of the moving average window.
            standard_deviation (int): Standard deviation threshold for outlier detection.
            product_column (str): Name of the column representing product groups.

        Returns:
            pandas.DataFrame: DataFrame with filled missing values.
        """
        df = self.transform_data()
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.sort_values('TimeStamp').reset_index()
        df = df.groupby([product_column, 'TimeStamp']).sum(['Quantity']).reset_index().drop(columns=['index'], errors='ignore')
        df['moving_average'] = df.groupby(product_column)['Quantity'].rolling(window=window_size).mean().reset_index(0, drop=True)
        
        # Calculate lower and upper bounds for replacement
        lower_bound = df['moving_average'] / standard_deviation
        upper_bound = df['moving_average'] * standard_deviation

        # Replace outliers with None
        df.loc[(df['Quantity'] < lower_bound) | (df['Quantity'] > upper_bound), 'Quantity'] = None

        # Interpolate missing values within each product group
        def interpolate_group(group):
            group['Quantity'].interpolate(method='linear', inplace=True)
            return group

        df = df.groupby(product_column).apply(interpolate_group).reset_index(drop=True)
        return df
    
    def outlier_correction_ml(self, product_column='ProductGroup'):
        """
        Perform outlier correction using machine learning techniques (Isolation Forest).

        Args:
            product_column (str): Name of the column representing product groups.

        Returns:
            pandas.DataFrame: DataFrame with outlier-corrected values.
        """
        df = self.fill_missing_value()
        scaler = StandardScaler()
        df['scaled_quantity'] = df.groupby(product_column)['Quantity'].transform(lambda x: np.concatenate(scaler.fit_transform(x.values.reshape(-1,1))))
        
        # Isolation Forest for outlier detection
        def isolation_forest(df):
            results = []
            unique_products = df[product_column].unique()

            for product in unique_products:
                product_data = df[df[product_column] == product]

                if ((product_data['scaled_quantity'].nunique() > 1) and 
                    (product_data.shape[0] > 100) and 
                    (np.mean(product_data['Quantity']) > 10)):
                    
                    model_data = product_data[['scaled_quantity']]
                    IF = IsolationForest(random_state=42, contamination=0.005, n_estimators=400, max_samples=0.8)
                    IF.fit(model_data)
                    score = IF.decision_function(model_data)
                    product_data['Score'] = score
                    results.append(product_data)

                else:
                    product_data['Score'] = 0
                    results.append(product_data)

            df_with_scores = pd.concat(results)
            return df_with_scores

        df = isolation_forest(df)

        # Replace outliers with None based on the anomaly score threshold
        df.loc[df['Score'] < -0.05, 'Quantity'] = None

        # Interpolate missing values within each product group
        def interpolate_group(group):
            group['Quantity'].interpolate(method='linear', inplace=True)
            return group

        df = df.groupby(product_column).apply(interpolate_group).reset_index(drop=True)

        return df
