import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd

# URL du fichier GeoJSON des frontières des pays européens
GEOJSON_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/EUR.geo.json"

# Données fictives pour les indices boursiers
indices_boursiers = {
    "France": {"CAC 40": 6000, "SBF 120": 4800},
    "Allemagne": {"DAX": 15000, "MDAX": 32000},
    "Portugal": {"PSI 20": 5200},
    "Suisse": {"SMI": 11000, "SPI": 14000}
}

def afficher_indice_pays(pays):
    st.title(f"Indices boursiers pour {pays}")
    if pays in indices_boursiers:
        data = indices_boursiers[pays]
        df = pd.DataFrame(list(data.items()), columns=["Indice", "Valeur"])
        st.table(df)
    else:
        st.write("Données non disponibles.")

def afficher_carte():
    m = folium.Map(location=[54, 15], zoom_start=4)

    # Charger les données GeoJSON
    geo_data = gpd.read_file(GEOJSON_URL)

    # Fonction pour ajouter les frontières des pays et les liens
    def style_function(feature):
        return {
            'fillOpacity': 0.1,
            'weight': 1,
            'color': 'black'
        }

    for _, row in geo_data.iterrows():
        country = row["name"]
        if country in indices_boursiers:
            folium.GeoJson(
                row["geometry"],
                style_function=style_function,
                tooltip=country,
                popup=f"<a href='?pays={country}'>{country}</a>"
            ).add_to(m)

    st_folium(m, width=700, height=500)

# Début de l'application Streamlit
st.set_page_config(page_title="Carte des Indices Boursiers", layout="wide")

pays_selectionne = st.experimental_get_query_params().get('pays', [None])[0]

if pays_selectionne:
    afficher_indice_pays(pays_selectionne)
else:
    st.title("Carte des Indices Boursiers en Europe")
    afficher_carte()
