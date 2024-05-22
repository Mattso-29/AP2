import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from folium.plugins import GeoJson

# Données fictives pour les indices boursiers
indices_boursiers = {
    "France": {"CAC 40": 6000, "SBF 120": 4800},
    "Allemagne": {"DAX": 15000, "MDAX": 32000},
    "Portugal": {"PSI 20": 5200},
    "Suisse": {"SMI": 11000, "SPI": 14000}
}

# GeoJSON data for countries (simplified for this example)
geojson_data = {
    "France": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[2.3522, 48.8566], [2.4, 48.8], [2.5, 48.9], [2.3522, 48.8566]]]}},
    "Allemagne": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[13.405, 52.52], [13.45, 52.5], [13.5, 52.55], [13.405, 52.52]]]}},
    "Portugal": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-9.139, 38.71667], [-9.2, 38.7], [-9.3, 38.8], [-9.139, 38.71667]]]}},
    "Suisse": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[7.44744, 46.94809], [7.5, 46.9], [7.6, 47], [7.44744, 46.94809]]]}}
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

    # Ajouter des polygones colorés pour chaque pays
    couleurs = {
        "France": "blue",
        "Allemagne": "green",
        "Portugal": "red",
        "Suisse": "yellow"
    }

    for pays, geojson in geojson_data.items():
        folium.GeoJson(
            geojson,
            style_function=lambda x, couleur=couleurs[pays]: {'color': couleur, 'fillColor': couleur, 'fillOpacity': 0.5},
            highlight_function=lambda x: {'weight': 3, 'color': 'black'},
            tooltip=folium.Tooltip(pays),
            popup=folium.Popup(f"<a href='?pays={pays}'>{pays}</a>")
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
