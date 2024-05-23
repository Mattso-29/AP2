import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Data_loader import load_all_data, prepare_macroeconomic_data, add_weekly_column
from map_loader import display_map

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

macro_data = prepare_macroeconomic_data()

# Add weekly data to country datasets
france = add_weekly_column(france, macro_data['bond_weekly'], 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
france = add_weekly_column(france, macro_data['bci_weekly'], 'FR SURVEY: BUSINESS CLIMATE FOR FRANCE NADJ', 'BCI')
france = add_weekly_column(france, macro_data['cci_weekly'], 'FR CONSUMER CONFIDENCE INDICATOR SADJ', 'CCI')
france = add_weekly_column(france, macro_data['gdp_weekly'], 'FRANCE GDP (CON) ', 'GDP')
france = add_weekly_column(france, macro_data['inflation_weekly'], 'FR INFLATION RATE ', 'Inflation')
france = add_weekly_column(france, macro_data['exchangerate_weekly'], 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
france = add_weekly_column(france, macro_data['unemployment_weekly'], 'FR ILO UNEMPLOYMENT RATE SADJ', 'Unemployment')
france = add_weekly_column(france, macro_data['gdp_weekly_normalized'], 'FRANCE GDP (CON) ', 'GDP(log)')

germany = add_weekly_column(germany, macro_data['bond_weekly'], 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
germany = add_weekly_column(germany, macro_data['bci_weekly'], 'BD TRADE & IND: BUS CLIMATE, INDEX, SA VOLA', 'BCI')
germany = add_weekly_column(germany, macro_data['cci_weekly'], 'BD CONSUMER CONFIDENCE INDICATOR - GERMANY SADJ', 'CCI')
germany = add_weekly_column(germany, macro_data['gdp_weekly'], 'Germany GDP CONA', 'GDP')
germany = add_weekly_column(germany, macro_data['inflation_weekly'], 'Germany INFLATION', 'Inflation')
germany = add_weekly_column(germany, macro_data['exchangerate_weekly'], 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
germany = add_weekly_column(germany, macro_data['unemployment_weekly'], 'BD UNEMPLOYMENT RATE - DEPENDENT CIVILIAN LABOUR FORCE NADJ', 'Unemployment')
germany = add_weekly_column(germany, macro_data['gdp_weekly_normalized'], 'Germany GDP CONA', 'GDP(log)')

switzerland = add_weekly_column(switzerland, macro_data['bond_weekly'], 'SW CONFEDERATION BOND YIELD - 10 YEARS NADJ', 'Bond_Yield')
switzerland = add_weekly_column(switzerland, macro_data['bci_weekly'], 'SW KOF IND. SURVEY: MACHINERY - BUSINESS CLIMATE(DISC.) NADJ', 'BCI')
switzerland = add_weekly_column(switzerland, macro_data['cci_weekly'], 'SW SECO CONSUMER CONFIDENCE INDICATOR SEASONAL ADJUSTED SADJ', 'CCI')
switzerland = add_weekly_column(switzerland, macro_data['gdp_weekly'], 'SW GDP (SA WDA) CONA', 'GDP')
switzerland = add_weekly_column(switzerland, macro_data['inflation_weekly'], 'SW ANNUAL INFLATION RATE NADJ', 'Inflation')
switzerland = add_weekly_column(switzerland, macro_data['exchangerate_weekly'], 'SW SWISS FRANCS TO USD NADJ', '1usd/chf')
switzerland = add_weekly_column(switzerland, macro_data['exchangerate_weekly'], 'SWISS FRANC TO EURO (WMR) - EXCHANGE RATE', '1eur/chf')
switzerland = add_weekly_column(switzerland, macro_data['unemployment_weekly'], 'SW UNEMPLOYMENT RATE (METHOD BREAK JAN 2014) NADJ', 'Unemployment')
switzerland = add_weekly_column(switzerland, macro_data['gdp_weekly_normalized'], 'SW GDP (SA WDA) CONA', 'GDP(log)')

portugal = add_weekly_column(portugal, macro_data['bond_weekly'], 'EM GOVERNMENT BOND YIELD - 10 YEAR NADJ', 'Bond_Yield')
portugal = add_weekly_column(portugal, macro_data['bci_weekly'], 'PT BUS SURVEY-MFG.: ECONOMIC CLIMATE INDICATOR (3MMA) NADJ', 'BCI')
portugal = add_weekly_column(portugal, macro_data['cci_weekly'], 'PT CONSUMER CONFIDENCE INDICATOR - PORTUGAL SADJ', 'CCI')
portugal = add_weekly_column(portugal, macro_data['gdp_weekly'], 'Portugal GDP CONA', 'GDP')
portugal = add_weekly_column(portugal, macro_data['inflation_weekly'], 'Portugal Inflation', 'Inflation')
portugal = add_weekly_column(portugal, macro_data['exchangerate_weekly'], 'EM U.S. $ TO 1 EURO (ECU PRIOR TO 1999) NADJ', '1euro/dollar')
portugal = add_weekly_column(portugal, macro_data['unemployment_weekly'], 'PT UNEMPLOYMENT RATE (METH. BREAK Q1.11) NADJ', 'Unemployment')
portugal = add_weekly_column(portugal, macro_data['gdp_weekly_normalized'], 'Portugal GDP CONA', 'GDP(log)')

country_data['France 🇫🇷'] = france
country_data['Germany 🇩🇪'] = germany
country_data['Switzerland 🇨🇭'] = switzerland
country_data['Portugal 🇵🇹'] = portugal

# Start Streamlit app
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
for country in country_data.keys():
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
        display_image_and_text(st.session_state['selected_country'])
        
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
            # Add content for regression analysis here

    with tabs[4]:
         st.write(f"Forecast for {st.session_state['selected_country']}")
        # Add content for forecast here
