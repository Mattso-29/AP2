import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_loader import load_all_data, stock_market_indices, geojson_data, center_coords

# Charger les données
france, germany, switzerland, portugal = load_all_data()
country_data = {
    'France 🇫🇷': france,
    'Germany 🇩🇪': germany,
    'Switzerland 🇨🇭': switzerland,
    'Portugal 🇵🇹': portugal
}

# Dictionnaire pour stocker les chemins des images et les descriptions
country_images_and_texts = {
    'France 🇫🇷': {
        'image': 'table events france.pdf',
        'text': 'Description of major macroeconomic events in France.'
    },
    'Germany 🇩🇪': {
        'image': 'table events germany.pdf',
        'text': 'Description of major macroeconomic events in Germany.'
    },
    'Switzerland 🇨🇭': {
        'image': 'table events switzerland.pdf',
        'text': 'Description of major macroeconomic events in Switzerland.'
    },
    'Portugal 🇵🇹': {
        'image': 'table events portugal.pdf',
        'text': 'Description of major macroeconomic events in Portugal.'
    }
}

def generer_graphique_indice(data, pays, columns, start_date, end_date, chart_type):
    data_filtered = data[start_date:end_date]
    
    plt.figure(figsize=(20, 12))
    if chart_type == 'Line':
        for column in columns:
            plt.plot(data_filtered.index, data_filtered[column], label=column)
    elif chart_type == 'Bar':
        data_filtered[columns].plot(kind='bar', ax=plt.gca())
    
    plt.title(f"Stock market index performance in {pays}")
    plt.xlabel("Date")
    plt.ylabel("Index value")
    plt.legend()
    st.pyplot(plt)
    plt.close()

def generer_heatmap_correlation(data, columns, start_date, end_date):
    data_filtered = data[start_date:end_date]
    correlation_matrix = data_filtered[columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    plt.close()

def afficher_indice_pays(pays):
    st.subheader(f"Stock market Index for {pays}")

    if pays in stock_market_indices:
        data = stock_market_indices[pays]['Sectors']
        indices = list(data.keys())
        
        df_indices = pd.DataFrame({
            'Index Name': [data[indice]['name'] for indice in indices],
            'Number of Companies': [data[indice]['companies'] for indice in indices]
        }, index=indices)
        
        st.write("### Index Study")
        st.table(df_indices)
        
        # Vérifier si les données du pays sont disponibles dans le dictionnaire
        if pays in country_data:
            country_df = country_data[pays]
            st.write("### Index Performance")
            
            # Options pour moduler les graphiques
            st.write("#### Modulate Graph")
            columns = st.multiselect("Select columns to display", country_df.columns.tolist(), default=country_df.columns.tolist())
            start_date = st.date_input("Start date", value=country_df.index.min(), min_value=country_df.index.min(), max_value=country_df.index.max())
            end_date = st.date_input("End date", value=country_df.index.max(), min_value=country_df.index.min(), max_value=country_df.index.max())
            chart_type = st.radio("Select chart type", ('Line', 'Bar'))
            
            generer_graphique_indice(country_df, pays, columns, start_date, end_date, chart_type)
            
            st.write("### Correlation Heatmap")
            st.write("#### Modulate Heatmap")
            heatmap_columns = st.multiselect("Select columns for heatmap", country_df.columns.tolist(), default=country_df.columns.tolist())
            generer_heatmap_correlation(country_df, heatmap_columns, start_date, end_date)
        else:
            st.write("No data available for the performance chart")
    else:
        st.write("No available")

def afficher_carte(pays_selectionne):
    if pays_selectionne and pays_selectionne in center_coords:
        center = center_coords[pays_selectionne]
        zoom_start = 6
    else:
        center = [48.8566, 2.3522]
        zoom_start = 4

    m = folium.Map(location=center, zoom_start=zoom_start, tiles='CartoDB positron')

    for pays, geojson in geojson_data.items():
        folium.GeoJson(
            geojson,
            style_function=lambda x, pays=pays: {
                'color': 'black' if pays == pays_selectionne else 'gray',
                'fillColor': 'black' if pays == pays_selectionne else 'darkgray',
                'fillOpacity': 0.7
            },
            highlight_function=lambda x: {'weight': 3, 'color': 'black'},
            tooltip=folium.Tooltip(pays)
        ).add_to(m)

    st_folium(m, width=1400, height=800)

# Début de l'application Streamlit
st.set_page_config(page_title="AP project", layout="wide")
# Inclure du CSS personnalisé pour styliser les boutons
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

# Gérer la sélection du pays avec un état
if 'pays_selectionne' not in st.session_state:
    st.session_state['pays_selectionne'] = None

# Affichage des boutons pour chaque pays
st.sidebar.title("Select a country")
for pays in stock_market_indices.keys():
    if st.sidebar.button(pays):
        st.session_state['pays_selectionne'] = pays

# Affichage de la carte
st.title("WORLD MAP 🗺")
afficher_carte(st.session_state['pays_selectionne'])

# Afficher les informations du pays sélectionné en dessous de la carte
if st.session_state['pays_selectionne']:
    # Ajouter les onglets
    tabs = st.tabs(["Country Analysis", "Major Macroeconomic Events", "Important Macroeconomic Variables", "Regression", "Forecast"])

    with tabs[0]:
        st.write(f"Analysis for {st.session_state['pays_selectionne']}")
        afficher_indice_pays(st.session_state['pays_selectionne'])

    with tabs[1]:
        st.write(f"Major macroeconomic events for {st.session_state['pays_selectionne']}")
        # Ajouter le contenu des événements macroéconomiques majeurs ici
        if st.session_state['pays_selectionne'] in country_images_and_texts:
            st.image(country_images_and_texts[st.session_state['pays_selectionne']]['image'], use_column_width=True)
            st.write(country_images_and_texts[st.session_state['pays_selectionne']]['text'])

    with tabs[2]:
        st.write(f"Important macroeconomic variables for {st.session_state['pays_selectionne']}")
        # Ajouter le contenu des variables macroéconomiques importantes ici

    with tabs[3]:
        st.write(f"Regression analysis for {st.session_state['pays_selectionne']}")
        # Ajouter le contenu de l'analyse de régression ici

    with tabs[4]:
        st.write(f"Forecast for {st.session_state['pays_selectionne']}")
        # Ajouter le contenu des prévisions ici

