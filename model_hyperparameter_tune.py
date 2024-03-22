from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.model_selection import ParameterGrid
import pandas as pd
import os
import matplotlib.pyplot as plt
from forecast import processed_data

class HyperParameterTuning:
    """
    A class for hyperparameter tuning using the Prophet forecasting model.

    Attributes:
        train (int): The number of training data points.
        window (int): The size of the rolling window for cross-validation.
        test (int): The number of test data points.
        data_path (str): The path to the directory containing data files.

    Methods:
        parameter_grid_prophet(): Performs hyperparameter tuning using the Prophet model.
    """

    def __init__(self, train, window, test, data_path=os.path.dirname(os.path.abspath(__file__))+'/'):
        """
        Initializes HyperParameterTuning class with parameters.

        Args:
            train (int): The number of training data points.
            window (int): The size of the rolling window for cross-validation.
            test (int): The number of test data points.
            data_path (str): The path to the directory containing data files.
        """
        self.train = train
        self.window = window
        self.test = test
        self.data_path = data_path

        if '\\' in self.data_path:
            self.data_path = self.data_path.replace('\\', '/')

    def parameter_grid_prophet(self, seasonality_mode=['additive', 'multiplicative'], changepoint_prior_scale=[0.2], n_changepoints=[50], element1=0, element2=2):
        """
        Performs hyperparameter tuning using the Prophet forecasting model.

        Args:
            seasonality_mode (list): List of seasonality modes to try.
            changepoint_prior_scale (list): List of changepoint prior scales to try.
            n_changepoints (list): List of numbers of changepoints to try.
            element1 (int): Start index for processing elements in the data.
            element2 (int): End index for processing elements in the data.

        Returns:
            pandas.DataFrame: DataFrame containing the best hyperparameters and RMSE for each product group.
            pandas.DataFrame: DataFrame containing cross-validation results.
            pandas.DataFrame: DataFrame containing performance metrics.
            pandas.DataFrame: DataFrame containing forecasted values.
        """
        df = pd.DataFrame(columns=['ProductGroup', 'best_params', 'best_rmse'])

        for element in processed_data['ProductGroup'].unique().tolist()[element1:element2]:
            data = processed_data[processed_data['ProductGroup'] == element].iloc[:, 1:3]
            data.columns = ['ds', 'y']

            # Define a grid of hyperparameters to search
            param_grid = {
                'seasonality_mode': seasonality_mode,
                'changepoint_prior_scale': changepoint_prior_scale,
                'n_changepoints': n_changepoints,
                # Add more hyperparameters and their values to search
            }

            # Create a list to store the best parameter combinations and their corresponding performance
            best_params = []
            best_rmse = float('inf')  # Initialize with a high RMSE value

            # Iterate over the parameter grid
            for params in ParameterGrid(param_grid):
                # Create and fit a Prophet model with the current parameter combination
                model = Prophet(**params, weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
                model.fit(data)  # Assuming you have prepared your data DataFrame

                future = model.make_future_dataframe(periods=90)
                forecast = model.predict(future)

                fig = model.plot(forecast)
                plt.title('{}'.format(element))
                fig = model.plot_components(forecast)
                plt.title('{}'.format(element))

                # Perform cross-validation
                df_cv = cross_validation(model, initial=self.train, period=self.window, horizon=self.test)  # Adjust as needed

                # Calculate RMSE from the cross-validation results
                df_p = performance_metrics(df_cv)
                rmse = df_p['rmse'].values[0]

                # Check if the current combination of hyperparameters is better than the previous best
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params.copy()

            # Print the best hyperparameters
            print("ProductGroup", element)
            print("Best hyperparameters:", best_params)
            print("Best RMSE:", best_rmse)
            df = df.append({
                "ProductGroup": element,
                "best_params": best_params,
                "best_rmse": best_rmse
                }, ignore_index=True)
            
        df.to_csv(self.data_path + 'ProphetHyperParameter/BestHyperParameter.csv', mode='a', sep=',', encoding='utf-8')

        return df, df_cv, df_p, forecast
