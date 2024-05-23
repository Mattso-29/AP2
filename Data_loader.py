import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def load_all_data():
    france = load_data('France copy.xlsx', [1, 4], '2000-01-01')
    germany = load_data('Allemagne copy.xlsx', [2], '2000-01-01')
    switzerland = load_data('Suisse copy.xlsx', [1], '2000-01-01')
    portugal = load_data('Portugal copy.xlsx', [], '2000-01-01')
    return france, germany, switzerland, portugal

def load_excel_with_dates(file_path, date_column):
    try:
        df = pd.read_excel(file_path, parse_dates=[date_column], index_col=date_column)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

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
