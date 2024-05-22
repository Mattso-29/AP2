import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

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
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=4)

    # Ajouter des marqueurs pour chaque pays
    pays_locations = {
        "France": [48.8566, 2.3522],
        "Allemagne": [52.52, 13.405],
        "Portugal": [38.71667, -9.139],
        "Suisse": [46.94809, 7.44744]
    }

    for pays, coords in pays_locations.items():
        folium.Marker(
            location=coords,
            popup=f"<a href='?pays={pays}'>{pays}</a>",
            tooltip=pays
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
