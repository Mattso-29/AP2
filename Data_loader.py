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
    france = load_data('France copy.xlsx', [1, 4], '2000-01-01')
    germany = load_data('Allemagne copy.xlsx', [2], '2000-01-01')
    switzerland = load_data('Suisse copy.xlsx', [1], '2000-01-01')
    portugal = load_data('Portugal copy.xlsx', [], '2000-01-01')
    return france, germany, switzerland, portugal

stock_market_indices = {
    "France 🇫🇷": {
        "Sectors": {
            "Technology": {"name": "CAC Technology Financial index (FRTEC)", "companies": 35},
            "Financials": {"name": "CAC Financials Financial index (FRFIN)", "companies": 39},
            "Industrials": {"name": "CAC Industrials Financial index (FRIN)", "companies": 60},
            "Telecom": {"name": "CAC Telecom (FRTEL)", "companies": 6}
        }
    },
    "Germany 🇩🇪": {
        "Sectors": {
            "Technology": {"name": "DAX Technology (CXPHX)", "companies": 18},
            "Financials": {"name": "DAX Financials (CXPVX)", "companies": 29},
            "Industrials": {"name": "DAX Industrials (CXPNX)", "companies": 64},
            "Telecom": {"name": "DAX Telecom (CXPTX)", "companies": 7}
        }
    },
    "Portugal 🇵🇹": {
        "Sectors": {
            "Technology": {"name": "PSI Technology (PTTEC)", "companies": 3},
            "Financials": {"name": "PSI Financials (PTFIN)", "companies": 2},
            "Industrials": {"name": "PSI Industrials (PTIN)", "companies": 6},
            "Telecom": {"name": "PSI Telecom (PTTEL)", "companies": 4}
        }
    },
    "Switzerland 🇨🇭": {
        "Sectors": {
            "Technology": {"name": "SWX Technology (C9500T)", "companies": 10},
            "Financials": {"name": "SWX Financials (C8700T)", "companies": 33},
            "Industrials": {"name": "SWX Industrials (C2700T)", "companies": 52},
            "Telecom": {"name": "SWX Telecom (C6500T)", "companies": 1}
        }
    }
}
