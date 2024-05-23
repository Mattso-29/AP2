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

# Functions to generate charts and heatmaps
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
        

def display_forecast(country):
    if country in country_images_and_texts and 'forecast' in country_images_and_texts[country]:
        sectors = ['Technology', 'Financials', 'Industrials', 'Telecom']
        selected_sector = st.selectbox(f"Select sector to display forecast", sectors)
        
        if selected_sector in country_images_and_texts[country]['forecast']:
            images_info = country_images_and_texts[country]['forecast'][selected_sector]

            sarima_info = next((info for info in images_info if 'sarima' in info['image'].lower()), None)
            prophet_info = next((info for info in images_info if 'prophet' in info['image'].lower()), None)
            
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
                {'image': 'forecast france tech1.png', 'text': 'Forecast 1 for Technology in France.'},
                {'image': 'forecast france tech2.png', 'text': 'Forecast 2 for Technology in France.'}
            ],
            'Financials': [
                {'image': 'forecast france fin1.png', 'text': 'Forecast 1 for Financials in France.'},
                {'image': 'forecast france fin2.png', 'text': 'Forecast 2 for Financials in France.'}
            ],
            'Industrials': [
                {'image': 'forecast france ind1.png', 'text': 'Forecast 1 for Industrials in France.'},
                {'image': 'forecast france ind2.png', 'text': 'Forecast 2 for Industrials in France.'}
            ],
            'Telecom': [
                {'image': 'forecast france tel1.png', 'text': 'Forecast 1 for Telecom in France.'},
                {'image': 'forecast france tel2.png', 'text': 'Forecast 2 for Telecom in France.'}
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
                {'image': 'forecast germany tech1.png', 'text': 'Forecast 1 for Technology in Germany.'},
                {'image': 'forecast germany tech2.png', 'text': 'Forecast 2 for Technology in Germany.'}
            ],
            'Financials': [
                {'image': 'forecast germany fin1.png', 'text': 'Forecast 1 for Financials in Germany.'},
                {'image': 'forecast germany fin2.png', 'text': 'Forecast 2 for Financials in Germany.'}
            ],
            'Industrials': [
                {'image': 'forecast germany ind1.png', 'text': 'Forecast 1 for Industrials in Germany.'},
                {'image': 'forecast germany ind2.png', 'text': 'Forecast 2 for Industrials in Germany.'}
            ],
            'Telecom': [
                {'image': 'forecast germany tel1.png', 'text': 'Forecast 1 for Telecom in Germany.'},
                {'image': 'forecast germany tel2.png', 'text': 'Forecast 2 for Telecom in Germany.'}
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
                {'image': 'forecast switzerland tech1.png', 'text': 'Forecast 1 for Technology in Switzerland.'},
                {'image': 'forecast switzerland tech2.png', 'text': 'Forecast 2 for Technology in Switzerland.'}
            ],
            'Financials': [
                {'image': 'forecast switzerland fin1.png', 'text': 'Forecast 1 for Financials in Switzerland.'},
                {'image': 'forecast switzerland fin2.png', 'text': 'Forecast 2 for Financials in Switzerland.'}
            ],
            'Industrials': [
                {'image': 'forecast switzerland ind1.png', 'text': 'Forecast 1 for Industrials in Switzerland.'},
                {'image': 'forecast switzerland ind2.png', 'text': 'Forecast 2 for Industrials in Switzerland.'}
            ],
            'Telecom': [
                {'image': 'forecast switzerland tel1.png', 'text': 'Forecast 1 for Telecom in Switzerland.'},
                {'image': 'forecast switzerland tel2.png', 'text': 'Forecast 2 for Telecom in Switzerland.'}
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
                {'image': 'forecast portugal tech1.png', 'text': 'Forecast 1 for Technology in Portugal.'},
                {'image': 'forecast portugal tech2.png', 'text': 'Forecast 2 for Technology in Portugal.'}
            ],
            'Financials': [
                {'image': 'forecast portugal fin1.png', 'text': 'Forecast 1 for Financials in Portugal.'},
                {'image': 'forecast portugal fin2.png', 'text': 'Forecast 2 for Financials in Portugal.'}
            ],
            'Industrials': [
                {'image': 'forecast portugal ind1.png', 'text': 'Forecast 1 for Industrials in Portugal.'},
                {'image': 'forecast portugal ind2.png', 'text': 'Forecast 2 for Industrials in Portugal.'}
            ],
            'Telecom': [
                {'image': 'forecast portugal tel1.png', 'text': 'Forecast 1 for Telecom in Portugal.'},
                {'image': 'forecast portugal tel2.png', 'text': 'Forecast 2 for Telecom in Portugal.'}
            ],
        }
    }
}

            
    
st.markdown("""
    <style>
    .stButton>button {
        display: block;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: darkgray;
    }
    </style>
    """, unsafe_allow_html=True)

if 'selected_country' not in st.session_state:
    st.session_state['selected_country'] = None

st.sidebar.title("Select a country")
for country in stock_market_indices.keys():
    if st.sidebar.button(country):
        st.session_state['selected_country'] = country

st.title("WORLD MAP 🗺")
display_map(st.session_state['selected_country'])

if st.session_state['selected_country']:
    tabs = st.tabs(["Country Analysis", "Major Macroeconomic Events", "Important Macroeconomic Variables", "Regression", "Forecast"])

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


