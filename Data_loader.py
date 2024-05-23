import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to load data and drop specified columns
def load_data(file_path, columns_to_drop, start_date):
    try:
        df = pd.read_excel(file_path)
        df.set_index(df.columns[0], inplace=True)
        df.index = df.index.astype(str)
        df.index = pd.to_datetime(df.index)
        df = df.drop(df.columns[columns_to_drop], axis=1)
        df = df.loc[start_date:]
        return df
    except Exception as e:
        return pd.DataFrame()

# Function to load all data for the specified countries
def load_all_data():
    france = load_data('France copy.xlsx', [1, 4], '2000-01-01')
    germany = load_data('Allemagne copy.xlsx', [2], '2000-01-01')
    switzerland = load_data('Suisse copy.xlsx', [1], '2000-01-01')
    portugal = load_data('Portugal copy.xlsx', [], '2000-01-01')
    return france, germany, switzerland, portugal

# Load macroeconomic variables with error handling
def load_excel_with_dates(file_path, date_column):
    try:
        df = pd.read_excel(file_path, parse_dates=[date_column], index_col=date_column)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def quarter_to_date(quarter):
    year = int(quarter.split()[1])
    q = quarter.split()[0]
    if q == 'Q1':
        return pd.Timestamp(f'{year}-01-01')
    elif q == 'Q2':
        return pd.Timestamp(f'{year}-04-01')
    elif q == 'Q3':
        return pd.Timestamp(f'{year}-07-01')
    elif q == 'Q4':
        return pd.Timestamp(f'{year}-10-01')
    else:
        raise ValueError(f"Unrecognized quarter: {quarter}")

def to_weekly(macro_df, method='ffill'):
    macro_df.index = pd.to_datetime(macro_df.index, errors='coerce')
    if method == 'ffill':
        weekly_df = macro_df.resample('W-SUN').ffill()
    elif method == 'interpolate':
        weekly_df = macro_df.resample('W-SUN').interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weekly_df

def add_weekly_column(country_df, weekly_df, weekly_column, new_column_name):
    selected_column = weekly_df[[weekly_column]].rename(columns={weekly_column: new_column_name})
    return country_df.join(selected_column, how='left')

def prepare_macroeconomic_data():
    bond = load_excel_with_dates('10Y Bond copy.xlsx', 0)
    bci = load_excel_with_dates('bci copy.xlsx', 0)
    cci = load_excel_with_dates('CCI copy.xlsx', 0)
    exchangerate = load_excel_with_dates('Exchange rate copy.xlsx', 0)
    gdp = load_excel_with_dates('GDP copy.xlsx', 0)
    inflation = load_excel_with_dates('Inflation copy.xlsx', 0)
    unemployment = load_excel_with_dates('unemployment copy.xlsx', 0)

    date_index = pd.Index([quarter_to_date(q) for q in cci.index])
    cci.index = date_index
    date_index = pd.Index([quarter_to_date(q) for q in unemployment.index])
    unemployment.index = date_index

    bond = bond.loc['2000-01-01':]
    numeric_columns = bond.select_dtypes(include=np.number).columns
    bond = bond[numeric_columns].apply(np.log1p)

    bci = bci.loc['2000-01-01':]
    scaler = MinMaxScaler()
    numeric_columns = bci.select_dtypes(include=np.number).columns
    bci[numeric_columns] = scaler.fit_transform(bci[numeric_columns])

    cci = cci.loc['2000-01-01':]
    scaler = MinMaxScaler()
    numeric_columns = cci.select_dtypes(include=np.number).columns
    cci[numeric_columns] = scaler.fit_transform(cci[numeric_columns])

    gdp = gdp.loc['2000-01-01':]
    numeric_columns = gdp.select_dtypes(include=np.number).columns
    gdp_normalized = gdp[numeric_columns].apply(np.log1p)

    exchangerate = exchangerate.loc['2000-01-01':]
    inflation = inflation.loc['2000-01-01':]
    inflation = inflation.drop(inflation.columns[[4]], axis=1)

    unemployment = unemployment.loc['2000-01-01':]
    scaler = MinMaxScaler()
    numeric_columns = unemployment.select_dtypes(include=np.number).columns
    unemployment[numeric_columns] = scaler.fit_transform(unemployment[numeric_columns])

    start_date = '2000-01-05'
    end_date = '2024-05-01'
    new_index = pd.date_range(start=start_date, end=end_date, freq='W-WED')

    bond_weekly = to_weekly(bond, method='ffill')
    bond_weekly.index = bond_weekly.index + pd.DateOffset(days=3)
    bond_weekly = bond_weekly.reindex(new_index)
    bond_weekly = bond_weekly.apply(pd.to_numeric, errors='coerce')
    bond_weekly.fillna(method='bfill', inplace=True)
    bond_weekly.interpolate(method='linear', inplace=True)

    bci_weekly = to_weekly(bci, method='ffill')
    bci_weekly.index = bci_weekly.index + pd.DateOffset(days=3)
    bci_weekly = bci_weekly.reindex(new_index)
    bci_weekly = bci_weekly.apply(pd.to_numeric, errors='coerce')
    bci_weekly.fillna(method='bfill', inplace=True)
    bci_weekly.interpolate(method='linear', inplace=True)

    cci_weekly = to_weekly(cci, method='ffill')
    cci_weekly.index = cci_weekly.index + pd.DateOffset(days=3)
    cci_weekly = cci_weekly.reindex(new_index)
    cci_weekly = cci_weekly.apply(pd.to_numeric, errors='coerce')
    cci_weekly.fillna(method='bfill', inplace=True)
    cci_weekly.interpolate(method='linear', inplace=True)

    gdp_weekly = to_weekly(gdp, method='ffill')
    gdp_weekly.index = gdp_weekly.index + pd.DateOffset(days=3)
    gdp_weekly = gdp_weekly.reindex(new_index)
    gdp_weekly = gdp_weekly.apply(pd.to_numeric, errors='coerce')
    gdp_weekly.fillna(method='bfill', inplace=True)
    gdp_weekly.interpolate(method='linear', inplace=True)

    gdp_weekly_normalized = to_weekly(gdp_normalized, method='ffill')
    gdp_weekly_normalized.index = gdp_weekly_normalized.index + pd.DateOffset(days=3)
    gdp_weekly_normalized = gdp_weekly_normalized.reindex(new_index)
    gdp_weekly_normalized = gdp_weekly_normalized.apply(pd.to_numeric, errors='coerce')
    gdp_weekly_normalized.fillna(method='bfill', inplace=True)
    gdp_weekly_normalized.interpolate(method='linear', inplace=True)

    inflation_weekly = to_weekly(inflation, method='ffill')
    inflation_weekly.index = inflation_weekly.index + pd.DateOffset(days=3)
    inflation_weekly = inflation_weekly.reindex(new_index)
    inflation_weekly = inflation_weekly.apply(pd.to_numeric, errors='coerce')
    inflation_weekly.fillna(method='bfill', inplace=True)
    inflation_weekly.interpolate(method='linear', inplace=True)

    exchangerate_weekly = to_weekly(exchangerate, method='ffill')
    exchangerate_weekly.index = exchangerate_weekly.index + pd.DateOffset(days=3)
    exchangerate_weekly = exchangerate_weekly.reindex(new_index)
    exchangerate_weekly = exchangerate_weekly.apply(pd.to_numeric, errors='coerce')
    exchangerate_weekly.fillna(method='bfill', inplace=True)
    exchangerate_weekly.interpolate(method='linear', inplace=True)

    unemployment_weekly = to_weekly(unemployment, method='ffill')
    unemployment_weekly.index = unemployment_weekly.index + pd.DateOffset(days=3)
    unemployment_weekly = unemployment_weekly.reindex(new_index)
    unemployment_weekly = unemployment_weekly.apply(pd.to_numeric, errors='coerce')
    unemployment_weekly.fillna(method='bfill', inplace=True)
    unemployment_weekly.interpolate(method='linear', inplace=True)

    return {
        'bond_weekly': bond_weekly,
        'bci_weekly': bci_weekly,
        'cci_weekly': cci_weekly,
        'gdp_weekly': gdp_weekly,
        'gdp_weekly_normalized': gdp_weekly_normalized,
        'inflation_weekly': inflation_weekly,
        'exchangerate_weekly': exchangerate_weekly,
        'unemployment_weekly': unemployment_weekly
    }

