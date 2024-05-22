import pandas as pd

def load_data(file_path, columns_to_drop, start_date):
    df = pd.read_excel(file_path)
    df.set_index(df.columns[0], inplace=True)
    df.index = df.index.astype(str)
    df.index = pd.to_datetime(df.index)
    df = df.drop(df.columns[columns_to_drop], axis=1)
    df = df.loc[start_date:]
    return df

def load_all_data():
    france = load_data('France.xlsx', [1, 4], '2000-01-01')
    germany = load_data('Allemagne.xlsx', [2], '2000-01-01')
    switzerland = load_data('Suisse.xlsx', [1], '2000-01-01')
    portugal = load_data('Portugal.xlsx', [], '2000-01-01')
    return france, germany, switzerland, portugal
