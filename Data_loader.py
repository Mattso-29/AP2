# data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

def prepare_data():
    # Load country data
    france, germany, switzerland, portugal = load_all_data()
    country_data = {
        'France 🇫🇷': france,
        'Germany 🇩🇪': germany,
        'Switzerland 🇨🇭': switzerland,
        'Portugal 🇵🇹': portugal
    }

    # Load macroeconomic data
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

    bond_weekly = to_weekly(bond, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    bci_weekly = to_weekly(bci, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    cci_weekly = to_weekly(cci, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    gdp_weekly = to_weekly(gdp, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    gdp_weekly_normalized = to_weekly(gdp_normalized, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    inflation_weekly = to_weekly(inflation, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    exchangerate_weekly = to_weekly(exchangerate, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')
    unemployment_weekly = to_weekly(unemployment, method='ffill').reindex(new_index).apply(pd.to_numeric, errors='coerce').fillna(method='bfill').interpolate(method='linear')

    france = add_weekly_column(france, bond_weekly, 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
    france = add_weekly_column(france, bci_weekly, 'FR SURVEY: BUSINESS CLIMATE FOR FRANCE NADJ', 'BCI')
    france = add_weekly_column(france, cci_weekly, 'FR CONSUMER CONFIDENCE INDICATOR SADJ', 'CCI')
    france = add_weekly_column(france, gdp_weekly, 'FRANCE GDP (CON) ', 'GDP')
    france = add_weekly_column(france, inflation_weekly, 'FR INFLATION RATE ', 'Inflation')
    france = add_weekly_column(france, exchangerate_weekly, 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
    france = add_weekly_column(france, unemployment_weekly, 'FR ILO UNEMPLOYMENT RATE SADJ', 'Unemployment')
    france = add_weekly_column(france, gdp_weekly_normalized, 'FRANCE GDP (CON) ', 'GDP(log)')

    germany = add_weekly_column(germany, bond_weekly, 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
    germany = add_weekly_column(germany, bci_weekly, 'BD TRADE & IND: BUS CLIMATE, INDEX, SA VOLA', 'BCI')
    germany = add_weekly_column(germany, cci_weekly, 'BD CONSUMER CONFIDENCE INDICATOR - GERMANY SADJ', 'CCI')
    germany = add_weekly_column(germany, gdp_weekly, 'Germany GDP CONA', 'GDP')
    germany = add_weekly_column(germany, inflation_weekly, 'Germany INFLATION', 'Inflation')
    germany = add_weekly_column(germany, exchangerate_weekly, 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
    germany = add_weekly_column(germany, unemployment_weekly, 'BD UNEMPLOYMENT RATE - DEPENDENT CIVILIAN LABOUR FORCE NADJ', 'Unemployment')
    germany = add_weekly_column(germany, gdp_weekly_normalized, 'Germany GDP CONA', 'GDP(log)')

    switzerland = add_weekly_column(switzerland, bond_weekly, 'SW CONFEDERATION BOND YIELD - 10 YEARS NADJ', 'Bond_Yield')
    switzerland = add_weekly_column(switzerland, bci_weekly, 'SW KOF IND. SURVEY: MACHINERY - BUSINESS CLIMATE(DISC.) NADJ', 'BCI')
    switzerland = add_weekly_column(switzerland, cci_weekly, 'SW SECO CONSUMER CONFIDENCE INDICATOR SEASONAL ADJUSTED SADJ', 'CCI')
    switzerland = add_weekly_column(switzerland, gdp_weekly, 'SW GDP (SA WDA) CONA', 'GDP')
    switzerland = add_weekly_column(switzerland, inflation_weekly, 'SW ANNUAL INFLATION RATE NADJ', 'Inflation')
    switzerland = add_weekly_column(switzerland, exchangerate_weekly, 'SW SWISS FRANCS TO USD NADJ', '1usd/chf')
    switzerland = add_weekly_column(switzerland, exchangerate_weekly, 'SWISS FRANC TO EURO (WMR) - EXCHANGE RATE', '1eur/chf')
    switzerland = add_weekly_column(switzerland, unemployment_weekly, 'SW UNEMPLOYMENT RATE (METHOD BREAK JAN 2014) NADJ', 'Unemployment')
    switzerland = add_weekly_column(switzerland, gdp_weekly_normalized, 'SW GDP (SA WDA) CONA', 'GDP(log)')

    portugal = add_weekly_column(portugal, bond_weekly, 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
    portugal = add_weekly_column(portugal, bci_weekly, 'PT BUS SURVEY-MFG.: ECONOMIC CLIMATE INDICATOR (3MMA) NADJ', 'BCI')
    portugal = add_weekly_column(portugal, cci_weekly, 'PT CONSUMER CONFIDENCE INDICATOR - PORTUGAL SADJ', 'CCI')
    portugal = add_weekly_column(portugal, gdp_weekly, 'Portugal GDP CONA', 'GDP')
    portugal = add_weekly_column(portugal, inflation_weekly, 'Portugal Inflation', 'Inflation')
    portugal = add_weekly_column(portugal, exchangerate_weekly, 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
    portugal = add_weekly_column(portugal, unemployment_weekly, 'PT UNEMPLOYMENT RATE (METH. BREAK Q1.11) NADJ', 'Unemployment')
    portugal = add_weekly_column(portugal, gdp_weekly_normalized, 'Portugal GDP CONA', 'GDP(log)')

    country_data['France 🇫🇷'] = france
    country_data['Germany 🇩🇪'] = germany
    country_data['Switzerland 🇨🇭'] = switzerland
    country_data['Portugal 🇵🇹'] = portugal

    return country_data


