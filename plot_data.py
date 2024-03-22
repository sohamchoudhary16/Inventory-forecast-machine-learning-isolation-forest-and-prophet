import plotly.express as px
from get_data import GetData
from preprocessing import Preprocessing

class plots:

    def __init__(self, df):
        self.df = df

    def timeseries_plot_plotly(self, x='TimeStamp', y='Quantity', product_column='ProductGroup'):   
        '''
        x: TimeStamp
        y: Quantity (default)
        product_column (color): CoreCockpitBezeichnungFilter (default)
        '''
        fig = px.line(self.df.groupby([product_column, 'TimeStamp']).sum(['Quantity']).reset_index(), x=x, y=y, color=product_column)
        fig.update_traces(textposition='top center')
        fig.update_layout(title_text='time series plot', title_x=0.5)
        fig.show()

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
    processed_data = preprocessing.transform_data()
    plots(processed_data).timeseries_plot_plotly(y = 'Quantity') 
    processed_data = preprocessing.fill_missing_value()
    plots(processed_data).timeseries_plot_plotly(y = 'Quantity')
    processed_data = preprocessing.outlier_correction_ml()
    plots(processed_data).timeseries_plot_plotly(y = 'Quantity')

    