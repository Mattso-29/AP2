import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_loader import load_all_data, stock_market_indices, geojson_data, center_coords

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
        'image': 'table_events_france.png',
        'text': 'Description of major macroeconomic events in France.'
    },
    'Germany 🇩🇪': {
        'image': 'table_events_germany.png',
        'text': 'Description of major macroeconomic events in Germany.'
    },
    'Switzerland 🇨🇭': {
        'image': 'table_events_switzerland.png',
        'text': 'Description of major macroeconomic events in Switzerland.'
    },
    'Portugal 🇵🇹': {
        'image': 'table_events_portugal.png',
        'text': 'Description of major macroeconomic events in Portugal.'
    }
}

def generate_index_chart(data, country, columns, start_date, end_date, chart_type):
    data_filtered = data[start_date:end_date]
    
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
    data_filtered = data[start_date:end_date]
    correlation_matrix = data_filtered[columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    plt.close()

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
            columns = st.multiselect("Select columns to display", country_df.columns.tolist(), default=country_df.columns.tolist())
            start_date = st.date_input("Start date", value=country_df.index.min(), min_value=country_df.index.min(), max_value=country_df.index.max())
            end_date = st.date_input("End date", value=country_df.index.max(), min_value=country_df.index.min(), max_value=country_df.index.max())
            chart_type = st.radio("Select chart type", ('Line', 'Bar'))
            
            generate_index_chart(country_df, country, columns, start_date, end_date, chart_type)
            
            st.write("### Correlation Heatmap")
            st.write("#### Modulate Heatmap")
            heatmap_columns = st.multiselect("Select columns for heatmap", country_df.columns.tolist(), default=country_df.columns.tolist())
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

def display_image_and_text(country):
    if country in country_images_and_texts:
        image_info = country_images_and_texts[country]
        image_path = image_info['image']
        try:
            st.image(image_path, use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
        st.write(image_info['text'])
    else:
        st.write("No data available")

# Start Streamlit app
st.set_page_config(page_title="AP Project", layout="wide")
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
  

