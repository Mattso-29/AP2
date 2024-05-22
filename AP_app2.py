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

# Coordonnées simplifiées des polygones pour chaque pays
geojson_data = {
    "France": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [2.5135730322461427, 51.14850617126183], [2.658422071960274, 50.796848049515745], 
                [3.123252179837022, 50.780363267614575], [3.588184441755686, 50.37899241800358], 
                [4.286022983425009, 49.907496649772554], [4.799221632515774, 49.985373033236385], 
                [5.674051954784867, 49.529483547557504], [5.897759230176376, 49.44266714130717], 
                [6.186320428094177, 49.463802802114515], [6.658229607783709, 49.20195831969157], 
                [8.099278598674856, 48.92221323447119], [7.593676385131062, 48.33301911070373], 
                [7.466759067422231, 47.62058197691181], [7.192202182655507, 47.44976552997099], 
                [6.736571079138088, 47.541801255882845], [6.768714490142179, 47.2877082383037], 
                [6.037388950228972, 46.72577919329534], [6.022609490593538, 46.272986554713075], 
                [6.500099724970454, 46.42967275652944], [6.843592970414562, 45.99114655210061], 
                [6.802355177445605, 45.70857982032867], [7.096652459347837, 45.33309937248707], 
                [6.749955275101711, 45.02851797136759], [7.007562290076663, 44.25476675066139], 
                [7.549596388386107, 44.12790110938481], [7.435184767291844, 43.69384491634918], 
                [6.529245232783068, 43.12889232031836], [4.556962517931396, 43.399650987311584], 
                [3.10041059735272, 43.075200507167125], [2.985998976258458, 42.47301504166986], 
                [1.8267932470871534, 42.34338471126569], [1.450982254803491, 42.61412697096747], 
                [0.7015906103638947, 42.795734361332606], [0.3380469091905802, 42.579546006839564], 
                [-1.502770961910471, 43.03401439063043], [-1.901351284177764, 43.42280202897834], 
                [-1.384225226234801, 44.02261037859017], [-1.193797573237362, 44.386964229556344], 
                [-1.733144837822503, 44.85491996705998], [-1.387163072052938, 45.62505008478871], 
                [-1.193797573237362, 45.45994971955622], [-2.225724249673845, 47.06436269793821], 
                [-2.963276129559609, 47.570326646507965], [-4.491554938159481, 47.95495433205642], 
                [-4.592349819344747, 48.68416046812695], [-3.295813971357745, 48.90169240985963], 
                [-1.6165107893849323, 48.64442129169458], [-1.9334940250633028, 49.77634186461576], 
                [-0.98946895995536, 49.347375800160876], [1.338761020522753, 49.04179991858197], 
                [1.93279677495198, 49.77634186461576], [2.5135730322461427, 51.14850617126183] ]] } },
           
"Allemagne": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [6.658230, 49.201958], [8.099279, 48.922213], [8.317301, 48.000779], [9.220000, 47.525000],
                [9.580000, 47.302487], [9.594226, 47.525058], [10.449000, 47.453000], [10.988000, 47.302000],
                [11.333333, 47.272697], [11.621900, 47.234900], [12.141500, 47.703400], [13.035500, 47.703400],
                [12.932627, 47.574800], [13.624300, 47.347400], [13.825300, 47.525700], [14.553300, 46.722000],
                [14.630000, 47.071500], [14.595800, 47.592200], [15.096100, 47.912300], [15.690800, 48.003300],
                [15.940900, 48.305300], [15.253100, 48.550900], [14.594100, 48.684800], [13.243357, 49.308434],
                [12.884103, 49.521096], [12.519581, 49.547415], [12.415191, 49.969121], [11.980166, 50.034305],
                [11.624033, 50.201992], [10.939467, 50.153601], [10.454140, 50.329413], [10.332752, 50.435146],
                [10.250487, 50.721722], [9.922837, 50.870391], [9.207140, 51.048353], [8.740000, 51.056000],
                [8.315300, 51.529700], [7.255180, 51.165696], [6.074183, 51.267258], [6.074183, 50.803721],
                [6.242751, 50.679700], [6.043073, 50.128052], [5.782417, 50.090327], [5.981052, 49.995185],
                [6.082177, 49.658624], [6.242751, 49.902225], [6.186320, 49.463803], [6.658230, 49.201958]
            ]]
        }
    },
    "Portugal": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-9.034818, 41.880570], [-8.671945, 42.134689], [-8.263857, 42.280469], [-8.013174, 41.790886],
                [-7.422513, 41.792075], [-7.251309, 41.918346], [-6.668606, 41.883387], [-6.389088, 41.381815],
                [-6.688129, 41.060944], [-7.026413, 40.631419], [-7.066592, 40.184524], [-7.374092, 39.711892],
                [-7.098037, 39.030073], [-7.498632, 38.484519], [-7.495437, 37.904098], [-7.855613, 37.546944],
                [-8.382816, 37.754467], [-8.898857, 38.268756], [-9.287464, 38.358486], [-9.526570, 38.737429],
                [-9.446989, 39.392066], [-9.048305, 39.755093], [-8.977353, 40.159306], [-8.768684, 40.760639],
                [-8.790853, 41.184334], [-8.990789, 41.543459], [-9.034818, 41.880570]
            ]]
        }
    },
    "Suisse": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [8.317301, 47.613580], [9.594226, 47.525058], [9.632932, 47.347601], [9.479970, 47.102810],
                [9.932448, 46.920728], [10.442701, 46.893546], [10.363378, 46.483571], [9.922837, 46.314899],
                [9.182882, 46.440215], [9.013602, 46.525058], [8.305346, 46.666667], [8.224157, 46.990528],
                [7.755379, 46.996682], [7.273851, 46.705129], [6.768714, 47.287708], [6.736571, 47.541801],
                [6.084666, 47.541801], [6.022609, 47.683723], [5.484203, 47.835241], [5.352254, 47.788575],
                [5.266007, 47.302000], [6.843593, 46.712939], [8.317301, 47.613580]
            ]]
        }
    }
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
