# Inventory Forecasting with Isolation Forest and Prophet

This project forecasts inventory demand using a two-step machine learning pipeline. First it cleans historical transaction data with Isolation Forest to remove anomalies, then it fits a Facebook Prophet model to forecast future demand by product group. The data comes from a Microsoft Dynamics 365 SQL Server.

---

## What it does

Raw inventory data is messy. Spikes from bulk orders, data entry errors, and one-off events all distort a forecast if you feed them in directly. This pipeline handles that by running the data through two stages:

**Stage 1: Cleaning**
A moving average pass flags obvious outliers first. Then Isolation Forest runs on each product group to catch subtler anomalies that the simple threshold misses. Flagged values get replaced with linearly interpolated estimates.

**Stage 2: Forecasting**
Prophet fits a time series model to the cleaned data. A grid search over seasonality mode, changepoint prior scale, and number of changepoints finds the best combination for each product group using cross-validation. Results get saved to a CSV for later use.

---

## Project structure

```
.
├── forecast.py                  # Entry point, wires everything together
├── get_data.py                  # Pulls data from SQL Server and reads product CSVs
├── preprocessing.py             # Cleans and denoises the time series
├── model_hyperparameter_tune.py # Prophet grid search with cross-validation
├── plot_data.py                 # Plotly charts for inspecting each pipeline stage
├── dataset_prep.ipynb           # Exploratory notebook
└── environment.yaml             # Conda environment (Python 3.7, Windows)
```

---

## Setup

The environment file is provided for reference but it is pinned to Windows and Python 3.7. A fresh conda or pip install with the core dependencies works fine on any platform:

```
prophet
scikit-learn
pandas
numpy
plotly
matplotlib
pyodbc
```

Install Prophet separately if needed since it requires a Stan backend:

```bash
pip install prophet
```

---

## Configuration

Open `forecast.py` and fill in your SQL Server credentials:

```python
GetDataInst = GetData(
    server   = 'your_server_name',
    database = 'your_database',
    username = 'your_username',
    password = 'your_password'
)
```

You also need two CSV files under `input_data/`:

- `Product.csv` with columns `Product`, `ProductGroup`
- `ProductIdList.csv` with columns `IdNumber`, `ProductGroup`

And a folder called `ProphetHyperParameter/` at the project root for the output CSV.

---

## Running it

```bash
python forecast.py
```

This will preprocess the data and run Prophet hyperparameter tuning on the first 20 product groups (controlled by `element1` and `element2` in `forecast.py`). Best parameters and RMSE scores get appended to `ProphetHyperParameter/BestHyperParameter.csv`.

To visualize the data at each preprocessing stage:

```bash
python plot_data.py
```

---

## How the outlier correction works

The `Preprocessing` class in `preprocessing.py` runs three steps in sequence:

1. `transform_data()` filters the transaction history to the relevant date range and joins product group labels
2. `fill_missing_value()` groups by product and day, computes a rolling average, and replaces values that fall way outside the moving average with interpolated ones
3. `outlier_correction_ml()` scales each product group and applies Isolation Forest with `contamination=0.005` to flag the most extreme points for replacement

Isolation Forest only runs on a product group if it has more than 100 observations and a mean quantity above 10. Smaller or low-volume groups skip the ML step and keep their scores at zero.

---

## Hyperparameter tuning

The `HyperParameterTuning` class tries every combination of:

- `seasonality_mode`: additive or multiplicative
- `changepoint_prior_scale`: controls how flexible the trend is
- `n_changepoints`: how many potential trend change points Prophet considers

For each combination it fits Prophet, runs cross-validation, and computes RMSE. The best combination per product group is saved to the output CSV.

The default call in `forecast.py` searches across a fairly wide grid:

```python
HyperParameterTuningInst = HyperParameterTuning(train='450 days', window='2 days', test='7 days')
df, df_cv, df_p, forecast = HyperParameterTuningInst.parameter_grid_prophet(
    seasonality_mode=['additive', 'multiplicative'],
    changepoint_prior_scale=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    n_changepoints=[100, 150, 200, 250],
    element1=0,
    element2=20
)
```

This is 64 combinations per product group, so expect it to run for a while on large datasets.

---

## Known issues

A few things to be aware of if you are running this on a newer Python or pandas version:

- `DataFrame.append()` was removed in pandas 2.0. The call in `model_hyperparameter_tune.py` needs replacing with `pd.concat()`.
- There is a circular import between `forecast.py` and `model_hyperparameter_tune.py` that will cause a crash. The fix is to pass `processed_data` as an argument to `parameter_grid_prophet()` instead of importing it at module level.
- `inplace=True` inside `groupby().apply()` in `preprocessing.py` does not always work as expected. Reassigning the result is safer.

---

## Requirements

- Python 3.8 or higher recommended
- A running Microsoft SQL Server instance with the inventory table
- Prophet 1.1+
- scikit-learn 1.0+
- pandas 1.3+ (note the pandas 2.0 issue above)
