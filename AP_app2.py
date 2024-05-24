import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from streamlit_folium import st_folium
import folium

from Data_loader import load_all_data, load_excel_with_dates, to_weekly, add_weekly_column
from map_loader import geojson_data, center_coords

# Set the page configuration
st.set_page_config(page_title="AP Project", layout="wide")

# Load data
france, germany, switzerland, portugal = load_all_data()
country_data = {
    'France 🇫🇷': france,
    'Germany 🇩🇪': germany,
    'Switzerland 🇨🇭': switzerland,
    'Portugal 🇵🇹': portugal
}


country_images_and_texts = {
    'France 🇫🇷': {
        'image': 'table events france.png',
        'text': 'Description of major macroeconomic events in France.'
    },
    'Germany 🇩🇪': {
        'image': 'table events germany.png',
        'text': 'Description of major macroeconomic events in Germany.'
    },
    'Switzerland 🇨🇭': {
        'image': 'table events switzerland.png',
        'text': 'Description of major macroeconomic events in Switzerland.'
    },
    'Portugal 🇵🇹': {
        'image': 'table events portugal.png',
        'text': 'Description of major macroeconomic events in Portugal.'
    }
}

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

def generate_index_chart(data, country, columns, start_date, end_date, chart_type):
    data_filtered = data.loc[start_date:end_date]
    
    plt.figure(figsize=(20, 12))
    if chart_type == 'Line':
        for column in columns:
            plt.plot(data_filtered.index, data_filtered[column], label=column)
    elif chart_type == 'Bar':
        data_filtered[columns].plot(kind='bar', ax=plt.gca())
    
    plt.title(f"Stock market index performance in {country}")
    plt.xlabel("Date")
    plt.ylabel("Index value")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def generate_correlation_heatmap(data, columns, start_date, end_date):
    available_columns = data.columns.tolist()
    selected_columns = [col for col in columns if col in available_columns]
    
    if not selected_columns:
        st.error("No columns selected for the heatmap are available in the data.")
        return
    
    data_filtered = data.loc[start_date:end_date, selected_columns]
    correlation_matrix = data_filtered.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# Functions to display content
def display_country_index(country):
    st.subheader(f"Stock market Index for {country}")

    if country in stock_market_indices:
        data = stock_market_indices[country]['Sectors']
        indices = list(data.keys())
        
        df_indices = pd.DataFrame({
            'Index Name': [data[indice]['name'] for indice in indices],
            'Number of Companies': [data[indice]['companies'] for indice in indices]
        }, index=indices)
        
        st.write("### Index Study")
        st.table(df_indices)
        
        if country in country_data:
            country_df = country_data[country]
            st.write("### Index Performance")
            
            st.write("#### Modulate Graph")
            columns = st.multiselect(f"Select columns to display ({country})", country_df.columns.tolist(), default=country_df.columns.tolist()[:4])  # Only the first 4 columns by default
            start_date = st.date_input(f"Start date ({country})", value=country_df.index.min(), min_value=country_df.index.min(), max_value=country_df.index.max())
            end_date = st.date_input(f"End date ({country})", value=country_df.index.max(), min_value=country_df.index.min(), max_value=country_df.index.max())
            chart_type = st.radio(f"Select chart type ({country})", ('Line', 'Bar'))
            
            generate_index_chart(country_df, country, columns, start_date, end_date, chart_type)
            
            st.write("### Correlation Heatmap")
            st.write("#### Modulate Heatmap")
            heatmap_columns = columns  # Use the selected columns for the heatmap
            generate_correlation_heatmap(country_df, heatmap_columns, start_date, end_date)
        else:
            st.write("No data available for the performance chart")
    else:
        st.write("No data available")

def display_map(selected_country):
    center = center_coords.get(selected_country, [48.8566, 2.3522])
    zoom_start = 6 if selected_country else 4

    m = folium.Map(location=center, zoom_start=zoom_start, tiles='CartoDB positron')

    for country, geojson in geojson_data.items():
        folium.GeoJson(
            geojson,
            style_function=lambda x, country=country: {
                'color': 'black' if country == selected_country else 'gray',
                'fillColor': 'black' if country == selected_country else 'darkgray',
                'fillOpacity': 0.7
            },
            highlight_function=lambda x: {'weight': 3, 'color': 'black'},
            tooltip=folium.Tooltip(country)
        ).add_to(m)

    st_folium(m, width=1400, height=800)

# Load macroeconomic variables
bond = load_excel_with_dates('10Y Bond copy.xlsx', 0)
bci = load_excel_with_dates('bci copy.xlsx', 0)
cci = load_excel_with_dates('CCI copy.xlsx', 0)
exchangerate = load_excel_with_dates('Exchange rate copy.xlsx', 0)
gdp = load_excel_with_dates('GDP copy.xlsx', 0)
inflation = load_excel_with_dates('Inflation copy.xlsx', 0)
unemployment = load_excel_with_dates('unemployment copy.xlsx', 0)

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
        raise ValueError(f"Trimestre non reconnu : {quarter}")

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

# Convertir les données macroéconomiques en données hebdomadaires
bond_weekly = to_weekly(bond, method='ffill')
bond_weekly.index = bond_weekly.index + pd.DateOffset(days=3)
start_date = '2000-01-05'
end_date = '2024-05-01'
new_index = pd.date_range(start=start_date, end=end_date, freq='W-WED')
bond_weekly = bond_weekly.reindex(new_index)
bond_weekly = bond_weekly[(bond_weekly.index >= start_date) & (bond_weekly.index <= end_date)]
bond_weekly = bond_weekly.apply(pd.to_numeric, errors='coerce')
bond_weekly.fillna(method='bfill', inplace=True)
bond_weekly.interpolate(method='linear', inplace=True)

bci_weekly = to_weekly(bci, method='ffill')
bci_weekly.index = bci_weekly.index + pd.DateOffset(days=3)
bci_weekly = bci_weekly.reindex(new_index)
bci_weekly = bci_weekly[(bci_weekly.index >= start_date) & (bci_weekly.index <= end_date)]
bci_weekly = bci_weekly.apply(pd.to_numeric, errors='coerce')
bci_weekly.fillna(method='bfill', inplace=True)
bci_weekly.interpolate(method='linear', inplace=True)

cci_weekly = to_weekly(cci, method='ffill')
cci_weekly.index = cci_weekly.index + pd.DateOffset(days=3)
cci_weekly = cci_weekly.reindex(new_index)
cci_weekly = cci_weekly[(cci_weekly.index >= start_date) & (cci_weekly.index <= end_date)]
cci_weekly = cci_weekly.apply(pd.to_numeric, errors='coerce')
cci_weekly.fillna(method='bfill', inplace=True)
cci_weekly.interpolate(method='linear', inplace=True)

gdp_weekly = to_weekly(gdp, method='ffill')
gdp_weekly.index = gdp_weekly.index + pd.DateOffset(days=3)
gdp_weekly = gdp_weekly.reindex(new_index)
gdp_weekly = gdp_weekly[(gdp_weekly.index >= start_date) & (gdp_weekly.index <= end_date)]
gdp_weekly = gdp_weekly.apply(pd.to_numeric, errors='coerce')
gdp_weekly.fillna(method='bfill', inplace=True)
gdp_weekly.interpolate(method='linear', inplace=True)

gdp_weekly_normalized = to_weekly(gdp_normalized, method='ffill')
gdp_weekly_normalized.index = gdp_weekly_normalized.index + pd.DateOffset(days=3)
gdp_weekly_normalized = gdp_weekly_normalized.reindex(new_index)
gdp_weekly_normalized = gdp_weekly_normalized[(gdp_weekly_normalized.index >= start_date) & (gdp_weekly.index <= end_date)]
gdp_weekly_normalized = gdp_weekly_normalized.apply(pd.to_numeric, errors='coerce')
gdp_weekly_normalized.fillna(method='bfill', inplace=True)
gdp_weekly_normalized.interpolate(method='linear', inplace=True)

inflation_weekly = to_weekly(inflation, method='ffill')
inflation_weekly.index = inflation_weekly.index + pd.DateOffset(days=3)
inflation_weekly = inflation_weekly.reindex(new_index)
inflation_weekly = inflation_weekly[(inflation_weekly.index >= start_date) & (inflation_weekly.index <= end_date)]
inflation_weekly = inflation_weekly.apply(pd.to_numeric, errors='coerce')
inflation_weekly.fillna(method='bfill', inplace=True)
inflation_weekly.interpolate(method='linear', inplace=True)

exchangerate_weekly = to_weekly(exchangerate, method='ffill')
exchangerate_weekly.index = exchangerate_weekly.index + pd.DateOffset(days=3)
exchangerate_weekly = exchangerate_weekly.reindex(new_index)
exchangerate_weekly = exchangerate_weekly[(exchangerate_weekly.index >= start_date) & (exchangerate_weekly.index <= end_date)]
exchangerate_weekly = exchangerate_weekly.apply(pd.to_numeric, errors='coerce')
exchangerate_weekly.fillna(method='bfill', inplace=True)
exchangerate_weekly.interpolate(method='linear', inplace=True)

unemployment_weekly = to_weekly(unemployment, method='ffill')
unemployment_weekly.index = unemployment_weekly.index + pd.DateOffset(days=3)
unemployment_weekly = unemployment_weekly.reindex(new_index)
unemployment_weekly = unemployment_weekly[(unemployment_weekly.index >= start_date) & (unemployment_weekly.index <= end_date)]
unemployment_weekly = unemployment_weekly.apply(pd.to_numeric, errors='coerce')
unemployment_weekly.fillna(method='bfill', inplace=True)
unemployment_weekly.interpolate(method='linear', inplace=True)

# Add weekly data to country datasets
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

# Add crises and stimulus policies to the data
crises_france = {'Dotcom bubble burst': ['2000-03-01', '2002-01-01'], 'Subprime crises': ['2008-01-01', '2009-01-01'], 'Covid 19': ['2020-01-01', '2021-01-01']}
stimulus_policies_france = {'LCME': ['2003-01-01', '2004-01-01'], 'Stimulus policy 2009': ['2009-01-01', '2010-01-01'], 'Stimulus policy Covid': ['2020-03-01', '2022-06-01']}

for crisis, (start_date, end_date) in crises_france.items():
    france[crisis] = ((france.index >= pd.to_datetime(start_date)) & (france.index <= pd.to_datetime(end_date))).astype(int)

for policy, (start_date, end_date) in stimulus_policies_france.items():
    france[policy] = ((france.index >= pd.to_datetime(start_date)) & (france.index <= pd.to_datetime(end_date))).astype(int)

crises_germany = {'Dotcom bubble burst': ['2000-03-01', '2002-01-01'], 'Subprime crises': ['2008-01-01', '2009-01-01'], 'Covid 19': ['2020-01-01', '2021-01-01']}
stimulus_policies_germany = {'Hartz and Agenda2010': ['2009-01-01', '2010-06-01'], 'Konjunkturpaket I': ['2008-11-01', '2011-01-01'], 'Konjunkturpaket II': ['2009-02-01', '2011-01-01'], 'Stimulus policy Covid': ['2020-03-01', '2023-01-01']}

for crisis, (start_date, end_date) in crises_germany.items():
    germany[crisis] = ((germany.index >= pd.to_datetime(start_date)) & (germany.index <= pd.to_datetime(end_date))).astype(int)

for policy, (start_date, end_date) in stimulus_policies_germany.items():
    germany[policy] = ((germany.index >= pd.to_datetime(start_date)) & (germany.index <= pd.to_datetime(end_date))).astype(int)

crises_switzerland = {'Dotcom bubble burst': ['2000-03-01', '2002-01-01'], 'Subprime crises': ['2008-01-01', '2009-01-01'], 'Covid 19': ['2020-01-01', '2021-01-01']}
stimulus_policies_switzerland = {'Stimulus policy 2001': ['2001-05-01', '2003-05-01'], 'Stimulus policy 2009': ['2009-01-01', '2010-01-01'], 'Stimulus policy Covid': ['2020-04-01', '2023-01-01']}

for crisis, (start_date, end_date) in crises_switzerland.items():
    switzerland[crisis] = ((switzerland.index >= pd.to_datetime(start_date)) & (switzerland.index <= pd.to_datetime(end_date))).astype(int)

for policy, (start_date, end_date) in stimulus_policies_switzerland.items():
    switzerland[policy] = ((switzerland.index >= pd.to_datetime(start_date)) & (switzerland.index <= pd.to_datetime(end_date))).astype(int)

crises_portugal = {'Sovereign debt crisis': ['2010-01-01', '2012-01-01'], 'Covid 19': ['2020-01-01', '2021-01-01']}
stimulus_policies_portugal = {'Stimulus debt crisis': ['2011-05-01', '2014-05-01'], 'Political Instability': ['2014-01-01', '2015-01-01'], 'Stimulus policy Covid': ['2020-03-01', '2022-06-01']}

for crisis, (start_date, end_date) in crises_portugal.items():
    portugal[crisis] = ((portugal.index >= pd.to_datetime(start_date)) & (portugal.index <= pd.to_datetime(end_date))).astype(int)

for policy, (start_date, end_date) in stimulus_policies_portugal.items():
    portugal[policy] = ((portugal.index >= pd.to_datetime(start_date)) & (portugal.index <= pd.to_datetime(end_date))).astype(int)

def display_image_and_text(country, section):
    if country in country_images_and_texts and section in country_images_and_texts[country]:
        images_info = country_images_and_texts[country][section]
        image_options = [info['image'] for info in images_info]
        selected_image = st.selectbox(f"Select image to display ({section})", image_options)
        image_info = next(info for info in images_info if info['image'] == selected_image)
        
        try:
            st.image(image_info['image'], use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
        st.write(image_info['text'])
    else:
        st.write("No data available")
        
def display_regression_model(country):
    if country in country_images_and_texts and 'regression' in country_images_and_texts[country]:
        models = ['Random Forest', 'Support Vector Regression']
        selected_model = st.selectbox(f"Select regression model", models)
        
        model_key = 'randomforest' if selected_model == 'Random Forest' else 'svr'
        image_info = country_images_and_texts[country]['regression'][model_key]
        
        if selected_model == 'Random Forest' and isinstance(image_info, list):
            cols = st.columns(2)
            texts = []
            for i, rf_data in enumerate(image_info):
                with cols[i % 2]:
                    try:
                        st.image(rf_data['image'], use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                    texts.append(rf_data['text'])
            # Combine texts into a single text block below the images
            combined_text = "\n\n".join(texts)
            st.write(combined_text)
        else:
            try:
                st.image(image_info['image'], use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
            st.write(image_info['text'])
    else:
        st.write("No data available")
        

import streamlit as st

def display_forecast(country):
    if country in country_images_and_texts and 'forecast' in country_images_and_texts[country]:
        sectors = ['Technology', 'Financials', 'Industrials', 'Telecom']
        selected_sector = st.selectbox(f"Select sector to display forecast", sectors)
        
        if selected_sector in country_images_and_texts[country]['forecast']:
            images_info = country_images_and_texts[country]['forecast'][selected_sector]

            sarima_info = next((info for info in images_info if 'sar' in info['image'].lower()), None)
            prophet_info = next((info for info in images_info if 'pro' in info['image'].lower()), None)
            
            if sarima_info:
                try:
                    st.image(sarima_info['image'], use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading SARIMA image: {e}")
                st.write(sarima_info['text'])
            
            if prophet_info:
                try:
                    st.image(prophet_info['image'], use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading Prophet image: {e}")
                st.write(prophet_info['text'])
        else:
            st.write("No data available for the selected sector")
    else:
        st.write("No data available")
        
country_images_and_texts = {
    'France 🇫🇷': {
        'events': [
            {'image': 'table events france.png', 'text': 'Description of macroeconomic events in France.'},
        ],
        'regression': {
            'randomforest': [
                {'image': 'RF1 france.png', 'text': 'Random Forest Regression analysis in France.'},
                {'image': 'RF2 france.png', 'text': ''}
            ],
            'svr': {'image': 'SVR france.png', 'text': 'SVR Regression analysis in France.'}
        },
        'forecast': {
            'Technology': [
                {'image': 'pro tech france.png', 'text': 'Forecast 1 for Technology in France.'},
                {'image': 'sar tech france.png', 'text': 'Forecast 2 for Technology in France.'}
            ],
            'Financials': [
                {'image': 'pro fin france.png', 'text': 'Forecast 1 for Financials in France.'},
                {'image': 'sar fin france.png', 'text': 'Forecast 2 for Financials in France.'}
            ],
            'Industrials': [
                {'image': 'pro ind france.png', 'text': 'Forecast 1 for Industrials in France.'},
                {'image': 'sar ind france.png', 'text': 'Forecast 2 for Industrials in France.'}
            ],
            'Telecom': [
                {'image': 'pro tel france.png', 'text': 'Forecast 1 for Telecom in France.'},
                {'image': 'sar tel france.png', 'text': 'Forecast 2 for Telecom in France.'}
            ],
        }
    },
    'Germany 🇩🇪': {
        'events': [
            {'image': 'table events germany.png', 'text': 'Description of macroeconomic events in Germany.'},  
        ],
        'regression': {
            'randomforest': [
                {'image': 'RF1 germany.png', 'text': 'Random Forest Regression analysis in Germany.'},
                {'image': 'RF2 germany.png', 'text': ''}
            ],
            'svr': {'image': 'SVR germany.png', 'text': 'SVR Regression analysis in Germany.'}
        },
        'forecast': {
            'Technology': [
                {'image': 'pro tech germany.png', 'text': 'Forecast 1 for Technology in Germany.'},
                {'image': 'sar tech germany.png', 'text': 'Forecast 2 for Technology in Germany.'}
            ],
            'Financials': [
                {'image': 'pro fin germany.png', 'text': 'Forecast 1 for Financials in Germany.'},
                {'image': 'sar fin germany.png', 'text': 'Forecast 2 for Financials in Germany.'}
            ],
            'Industrials': [
                {'image': 'pro ind germany.png', 'text': 'Forecast 1 for Industrials in Germany.'},
                {'image': 'sar ind germany.png', 'text': 'Forecast 2 for Industrials in Germany.'}
            ],
            'Telecom': [
                {'image': 'pro tel germany.png', 'text': 'Forecast 1 for Telecom in Germany.'},
                {'image': 'sar tel germany.png', 'text': 'Forecast 2 for Telecom in Germany.'}
            ],
        }
    },
    'Switzerland 🇨🇭': {
        'events': [
            {'image': 'table events switzerland.png', 'text': 'Description of macroeconomic events in Switzerland.'},
        ],
        'regression': {
            'randomforest': [
                {'image': 'RF1 switzerland.png', 'text': 'Random Forest Regression analysis in Switzerland.'},
                {'image': 'RF2 switzerland.png', 'text': ''}
            ],
            'svr': {'image': 'SVR switzerland.png', 'text': 'SVR Regression analysis in Switzerland.'}
        },
        'forecast': {
            'Technology': [
                {'image': 'pro tech switzerland.png', 'text': 'Forecast 1 for Technology in Switzerland.'},
                {'image': 'sar tech switzerland.png', 'text': 'Forecast 2 for Technology in Switzerland.'}
            ],
            'Financials': [
                {'image': 'pro fin switzerland.png', 'text': 'Forecast 1 for Financials in Switzerland.'},
                {'image': 'sar fin switzerland.png', 'text': 'Forecast 2 for Financials in Switzerland.'}
            ],
            'Industrials': [
                {'image': 'pro ind switzerland.png', 'text': 'Forecast 1 for Industrials in Switzerland.'},
                {'image': 'sar ind switzerland.png', 'text': 'Forecast 2 for Industrials in Switzerland.'}
            ],
            'Telecom': [
                {'image': 'pro tel switzerland.png', 'text': 'Forecast 1 for Telecom in Switzerland.'},
                {'image': 'sar tel switzerland.png', 'text': 'Forecast 2 for Telecom in Switzerland.'}
            ],
        }
    },
    'Portugal 🇵🇹': {
        'events': [
            {'image': 'table events portugal.png', 'text': 'Description of macroeconomic events in Portugal.'},
        ],
        'regression': {
            'randomforest': [
                {'image': 'RF1 portugal.png', 'text': 'Random Forest Regression analysis in Portugal.'},
                {'image': 'RF2 portugal.png', 'text': ''}
            ],
            'svr': {'image': 'SVR portugal.png', 'text': 'SVR Regression analysis in Portugal.'}
        },
        'forecast': {
            'Technology': [
                {'image': 'pro tech portugal.png', 'text': 'Forecast 1 for Technology in Portugal.'},
                {'image': 'sar tech portugal.png', 'text': 'Forecast 2 for Technology in Portugal.'}
            ],
            'Financials': [
                {'image': 'pro fin portugal.png', 'text': 'Forecast 1 for Financials in Portugal.'},
                {'image': 'sar fin portugal.png', 'text': 'Forecast 2 for Financials in Portugal.'}
            ],
            'Industrials': [
                {'image': 'pro ind portugal.png', 'text': 'Forecast 1 for Industrials in Portugal.'},
                {'image': 'sar ind portugal.png', 'text': 'Forecast 2 for Industrials in Portugal.'}
            ],
            'Telecom': [
                {'image': 'pro tel portugal.png', 'text': 'Forecast 1 for Telecom in Portugal.'},
                {'image': 'sar tel portugal.png', 'text': 'Forecast 2 for Telecom in Portugal.'}
            ],
        }
    }
}

            
    
st.markdown("""
    <style>
    /* General styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar {
        background-color: #333;
        color: white;
    }
    .stSidebar .stButton>button {
        background-color: #555;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        width: 100%;
    }
    .stSidebar .stButton>button:hover {
        background-color: #777;
    }
    .stTabs .stTab {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTabs .stTab:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .stMarkdown {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
if 'selected_country' not in st.session_state:
    st.session_state['selected_country'] = None

st.sidebar.title("Select a country")
for country in stock_market_indices.keys():
    if st.sidebar.button(country):
        st.session_state['selected_country'] = country

investment_advice = {
    'France 🇫🇷': {
        'Technology': """
        **Analysis:**
        - Historical Performance: The Technology sector showed a significant decline around 2000-2002, followed by a long period of stability and recent growth from around 2015 onwards.
        - Forecast: Projections indicate a steady increase, though with high uncertainty reflected in the wide forecast range.
        
        **Political and Economic Outlook:**
        - Government Support: France is likely to continue investing in technology, given its strategic importance for economic competitiveness.
        - EU Regulations: Stricter EU regulations on data privacy and technology could impact growth but also stabilize the market by creating trust.
        
        **Investment Advice:**
        - Long-Term Investment: The Technology sector appears promising for long-term investment, considering the growing importance of digital transformation and AI.
        - Monitor Regulations: Keep an eye on EU regulations as they can impact the sector’s profitability.
        """,
        'Financials': """
        **Analysis:**
        - Historical Performance: The Financials sector experienced significant volatility, particularly around the 2008 financial crisis.
        - Forecast: The forecast suggests moderate growth with a fair amount of uncertainty.
        
        **Political and Economic Outlook:**
        - Regulatory Environment: Financial regulations in the EU are becoming stricter, which may limit aggressive growth but enhance stability.
        - Interest Rates: Changes in ECB policies on interest rates will significantly impact this sector.
        
        **Investment Advice:**
        - Moderate Growth: Expect moderate returns; financial stability measures and interest rates will be key determinants.
        - Diversify Within Sector: Diversifying within the financial sector (e.g., insurance, fintech) could reduce risk.
        """,
        'Industrials': """
        **Analysis:**
        - Historical Performance: This sector has shown steady growth with some cyclical downturns.
        - Forecast: Predictions show continued growth with some volatility, indicating a positive trend overall.
        
        **Political and Economic Outlook:**
        - Green Transition: The push towards green technologies and renewable energy is likely to bolster the Industrials sector.
        - Infrastructure Projects: Government spending on infrastructure projects will support this sector.
        
        **Investment Advice:**
        - Growth Potential: Given the focus on green technology and infrastructure, Industrials present a solid investment opportunity.
        - Cyclical Nature: Be prepared for some cyclical downturns; consider a diversified portfolio to mitigate risks.
        """,
        'Telecom': """
        **Analysis:**
        - Historical Performance: The Telecom sector has shown a long-term decline with some stabilization in recent years.
        - Forecast: Projections indicate very modest growth with high uncertainty, suggesting a stagnant market.
        
        **Political and Economic Outlook:**
        - 5G Rollout: Investment in 5G technology could provide growth opportunities.
        - Competition: High competition within the sector may suppress profitability.
        
        **Investment Advice:**
        - Selective Investment: Focus on companies investing in new technologies like 5G.
        - Short to Medium Term: Given the high uncertainty, a short to medium-term investment horizon may be prudent.
        """
    },
    'Germany 🇩🇪': {
        'Technology': """
        **Analysis:**
        - Historical Performance: The Technology sector in Germany has shown a steady increase since 2012 after a period of volatility.
        - Forecast: The projections indicate continued growth, with a moderate level of uncertainty.
        
        **Political and Economic Outlook:**
        - Government Support: The German government is heavily investing in digitalization and Industry 4.0, which will drive growth in the technology sector.
        - EU Regulations: Similar to France, EU regulations on data privacy and technology can influence the sector but are likely to create a stable market environment.
        
        **Investment Advice:**
        - Long-Term Growth: The Technology sector is expected to benefit from ongoing digital transformation initiatives, making it a strong candidate for long-term investment.
        - Monitor Innovation: Focus on companies that are leading in innovation and adapting to new regulations.
        """,
        'Financials': """
        **Analysis:**
        - Historical Performance: The Financials sector has been highly volatile, especially around the global financial crisis and recent economic uncertainties.
        - Forecast: The forecast suggests moderate growth with substantial uncertainty.
        
        **Political and Economic Outlook:**
        - Regulatory Environment: Stricter financial regulations in the EU could limit aggressive growth but enhance market stability.
        - Interest Rates: Changes in ECB policies on interest rates will directly affect the sector’s performance.
        
        **Investment Advice:**
        - Moderate Returns: Expect moderate growth; the sector offers stability, but regulatory and interest rate changes will be key factors.
        - Diversify Investments: Diversify within financial services to manage risk effectively.
        """,
        'Industrials': """
        **Analysis:**
        - Historical Performance: The Industrials sector has shown robust growth with cyclical downturns, reflecting the overall economic cycles.
        - Forecast: The forecast shows a continued upward trend with some expected volatility.
        
        **Political and Economic Outlook:**
        - Green Transition: The push for green technology and sustainable industrial practices is expected to support growth in the industrial sector.
        - Export Dependency: Germany's industrial sector is highly dependent on exports, which means global economic conditions will have a significant impact.
        
        **Investment Advice:**
        - Solid Growth: Investing in Industrials offers strong growth potential, particularly in green technologies and sustainable industries.
        - Global Economic Watch: Keep an eye on global economic conditions and trade policies as they can significantly impact this sector.
        """,
        'Telecom': """
        **Analysis:**
        - Historical Performance: The Telecom sector has shown a significant recovery since 2016 after a long period of stagnation.
        - Forecast: Projections indicate steady growth, with relatively lower uncertainty compared to other sectors.
        
        **Political and Economic Outlook:**
        - 5G Rollout: Investment in 5G infrastructure will drive growth and create new opportunities within the Telecom sector.
        - Market Competition: High competition in the telecom market might limit profit margins but will push innovation and customer-focused services.
        
        **Investment Advice:**
        - Growth in Infrastructure: Telecom presents a good investment opportunity, particularly with the ongoing 5G rollout.
        - Competitive Edge: Focus on companies that are leading in 5G and technological advancements.
        """
    },
    'Switzerland 🇨🇭': {
        'Technology': """
        **Analysis:**
        - Historical Performance: The Technology sector in Switzerland has shown significant volatility but has demonstrated strong growth in recent years.
        - Forecast: The projections indicate moderate growth with a moderate level of uncertainty.
        
        **Political and Economic Outlook:**
        - Innovation Hub: Switzerland is known for its innovation and investment in technology, providing a robust foundation for growth.
        - Regulatory Environment: The regulatory environment in Switzerland is favorable for technology companies, encouraging innovation and development.
        
        **Investment Advice:**
        - Long-Term Investment: The Technology sector is expected to continue benefiting from Switzerland’s strong innovation culture, making it a solid candidate for long-term investment.
        - Focus on Innovation: Prioritize companies that are at the forefront of technological advancements and innovations.
        """,
        'Financials': """
        **Analysis:**
        - Historical Performance: The Financials sector has experienced significant volatility, particularly around global financial crises.
        - Forecast: The forecast suggests moderate growth with substantial uncertainty.
        
        **Political and Economic Outlook:**
        - Regulatory Stability: Switzerland’s financial sector benefits from a stable regulatory environment, which enhances market confidence.
        - Global Financial Hub: Switzerland’s position as a global financial hub provides a stable foundation for growth.
        
        **Investment Advice:**
        - Moderate Returns: Expect moderate growth; the sector offers stability, but global financial conditions will be key factors.
        - Diversify Investments: Diversify within financial services to manage risk effectively.
        """,
        'Industrials': """
        **Analysis:**
        - Historical Performance: The Industrials sector has shown consistent growth with cyclical downturns, reflecting the overall economic cycles.
        - Forecast: The forecast shows a continued upward trend with some expected volatility.
        
        **Political and Economic Outlook:**
        - Sustainability Initiatives: Switzerland’s focus on sustainability and green technologies is expected to support growth in the industrial sector.
        - Export Dependency: Similar to Germany, Switzerland’s industrial sector is heavily reliant on exports, making it sensitive to global economic conditions.
        
        **Investment Advice:**
        - Solid Growth Potential: Investing in Industrials offers strong growth potential, particularly in green technologies and sustainable industries.
        - Global Economic Sensitivity: Keep an eye on global economic conditions and trade policies as they can significantly impact this sector.
        """,
        'Telecom': """
        **Analysis:**
        - Historical Performance: The Telecom sector has shown a steady recovery since 2016 after a long period of stagnation.
        - Forecast: Projections indicate steady growth with relatively lower uncertainty compared to other sectors.
        
        **Political and Economic Outlook:**
        - Technological Advancements: Investment in new technologies like 5G will drive growth and create new opportunities within the Telecom sector.
        - Competitive Market: High competition in the telecom market might limit profit margins but will push innovation and customer-focused services.
        
        **Investment Advice:**
        - Growth in Infrastructure: Telecom presents a good investment opportunity, particularly with the ongoing advancements in 5G technology.
        - Competitive Edge: Focus on companies that are leading in technological advancements and customer service.
        """
    },
    'Portugal 🇵🇹': {
        'Technology': """
        **Analysis:**
        The technology sector in Portugal has been expanding with significant investments in startups and digital innovation. The forecast indicates robust growth driven by increasing demand for digital services and technological advancements.
        
        **Political and Economic Outlook:**
        Portugal has been supportive of technological advancements with policies encouraging innovation and entrepreneurship. The global shift towards digitalization provides a favorable environment for tech investments.
        
        **Investment Advice:**
        Investing in the technology sector is highly recommended due to its strong growth potential. Focus on companies involved in software development, digital services, and innovative technologies. Startups and growth-stage tech companies present attractive investment opportunities.
        """,
        'Financials': """
        **Analysis:**
        The financial sector in Portugal has faced challenges, particularly during the European debt crisis. However, it has been stabilizing with regulatory reforms and improved banking practices. The forecast suggests a stable outlook with moderate growth.
        
        **Political and Economic Outlook:**
        Regulatory reforms and EU support have strengthened the financial sector. Continued political stability and economic reforms will be essential. The performance of the Eurozone and global financial markets will also influence this sector.
        
        **Investment Advice:**
        Investors should adopt a cautious approach with selective investments in well-capitalized banks and financial institutions. Consider financial services companies with a strong digital presence and innovative financial products.
        """,
        'Industrials': """
        **Analysis:**
        Portugal's industrial sector has shown a steady recovery post-2008 financial crisis with recent growth attributed to improvements in manufacturing and exports. The sector has been moderately volatile but shows a positive trend in recent years.
        
        **Political and Economic Outlook:**
        Portugal's economic policies have been favorable towards industrial growth with a focus on innovation and technology integration. However, political stability and economic reforms are crucial to maintain investor confidence. The European Union's policies and global economic conditions will also impact the sector.
        
        **Investment Advice:**
        Given the steady growth and positive outlook, a moderate investment in Portugal's industrial sector is recommended. Look for companies with strong export capabilities and innovative practices. Diversification within the industrial sub-sectors can mitigate risks.
        """,
        'Telecom': """
        **Analysis:**
        The telecom sector has experienced moderate growth, with fluctuations reflecting global trends. The forecast suggests a stable outlook with gradual growth as the sector adapts to new technologies and consumer demands.
        
        **Political and Economic Outlook:**
        Regulatory support and infrastructure investments are critical for the telecom sector. The government’s focus on improving digital infrastructure and connectivity will drive growth. However, global market trends and technological disruptions need to be considered.
        
        **Investment Advice:**
        Investments in the telecom sector should be focused on companies with strong infrastructure and the ability to adapt to new technologies like 5G. Consider investing in telecom companies with a diverse service portfolio and strong market presence.
        """
    }
}

def display_investment_advice(country):
    if country in investment_advice:
        st.subheader(f"Investment Advice for {country}")

        sectors = list(investment_advice[country].keys())
        selected_sector = st.selectbox(f"Select sector for investment advice", sectors)

        advice = investment_advice[country][selected_sector]
        
        st.write(advice)
    else:
        st.write("No investment advice available")




if __name__ == "__main__":
    # Your existing Streamlit code to run the app
    st.title("WORLD MAP 🗺")
    display_map(st.session_state['selected_country'])

    if st.session_state['selected_country']:
        tabs = st.tabs(["Country Analysis", "Major Macroeconomic Events", "Important Macroeconomic Variables", "Regression", "Forecast", "Investment Advice"])

        with tabs[0]:
            st.write(f"Analysis for {st.session_state['selected_country']}")
            display_country_index(st.session_state['selected_country'])

        with tabs[1]:
            st.write(f"Major macroeconomic events for {st.session_state['selected_country']}")
            display_image_and_text(st.session_state['selected_country'], 'events')

        with tabs[2]:
            st.write(f"Important macroeconomic variables for {st.session_state['selected_country']}")

            country_df = country_data[st.session_state['selected_country']]
            columns = country_df.columns.tolist()

            if st.session_state['selected_country'] == "Switzerland 🇨🇭":
                combined_columns = [
                    'SWX TECHNOLOGY - PRICE INDEX', 
                    'SWX FINANCIAL SVS - PRICE INDEX', 
                    'SWX INDS GDS & SVS - PRICE INDEX', 
                    'SWX TELECOM - PRICE INDEX', 
                    'Bond_Yield', 
                    'BCI', 
                    'CCI', 
                    'GDP', 
                    'Inflation', 
                    '1usd/chf', 
                    '1eur/chf', 
                    'Unemployment', 
                    'GDP(log)'
                ]
            else:
                combined_columns = list(set(columns[:4] + [
                    'Bond_Yield', 
                    'BCI', 
                    'CCI', 
                    'GDP', 
                    'Inflation', 
                    '1euro/dollar', 
                    'Unemployment', 
                    'GDP(log)'
                ]))

            valid_columns = [col for col in combined_columns if col in country_df.columns]

            heatmap_columns = st.multiselect(f"Select columns for heatmap ({st.session_state['selected_country']} - macroeconomic)", valid_columns, default=valid_columns)

            if heatmap_columns:
                generate_correlation_heatmap(country_df, heatmap_columns, start_date, end_date)
            else:
                st.write("No valid columns selected for heatmap.")

        with tabs[3]:
            st.write(f"Regression analysis for {st.session_state['selected_country']}")
            display_regression_model(st.session_state['selected_country'])

        with tabs[4]:
            st.write(f"Forecast for {st.session_state['selected_country']}")
            display_forecast(st.session_state['selected_country'])

        with tabs[5]:
            st.write(f"Investment advice for {st.session_state['selected_country']}")
            display_investment_advice(st.session_state['selected_country'])
