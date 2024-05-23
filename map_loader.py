import folium
from streamlit_folium import st_folium

# Coordonnées centrales pour zoomer sur chaque pays (décalées vers la droite)
center_coords = {
    "France 🇫🇷": [46.603354, 3.888334], 
    "Germany 🇩🇪": [51.165691, 12.451526], 
    "Portugal 🇵🇹": [39.399872, -6.224454],  
    "Switzerland 🇨🇭": [46.818188, 10.227512]  
}

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
