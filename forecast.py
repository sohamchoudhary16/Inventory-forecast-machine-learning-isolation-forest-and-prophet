
from get_data import GetData
from preprocessing import Preprocessing
from model_hyperparameter_tune import HyperParameterTuning

if __name__ == '__main__':

# Create an instance of CoreConn

    GetDataInst = GetData(
        server = '<your_SQL_server_name>',
        database = '<your_SQL_server_database>',
        username = '<your_username>',
        password = '<your_password>'
    )

    # Create an instance of Preprocessing and pass the transform_data function
    preprocessing = Preprocessing(GetDataInst.db_conn(), GetDataInst.products_id(), GetDataInst.products())
    processed_data = preprocessing.outlier_correction_ml()

    HyperParameterTuningInst = HyperParameterTuning(train='450 days', window='2 days', test='7 days')

    df, df_cv, df_p, forecast =  HyperParameterTuningInst.parameter_grid_prophet(seasonality_mode = ['additive', 'multiplicative'], changepoint_prior_scale=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], n_changepoints=[100, 150, 200, 250], element1=0, element2=20)
      