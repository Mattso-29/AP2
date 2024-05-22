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
    [2.5160965424793744, 51.062061390974804],
    [1.6689832206209871, 50.91957528888247],
    [1.5991928340295374, 50.316619680545585],
    [1.355391926990535, 50.043808424074854],
    [0.718745404110706, 49.839243995462965],
    [0.15341927383360598, 49.66326687140614],
    [0.17334515656585836, 49.37999880054517],
    [-0.6231793025797856, 49.299669326118334],
    [-1.18124597958942, 49.35714016316925],
    [-1.3258795494349158, 49.63695494664097],
    [-1.8220174041476582, 49.6946512568027],
    [-1.6322226528700128, 49.2516825505042],
    [-1.5083797632303515, 48.73319108231888],
    [-1.3558083160947376, 48.574752026080375],
    [-1.9589930704486846, 48.6708893732104],
    [-2.564406624251859, 48.55502686138962],
    [-3.079175035062491, 48.78535120554639],
    [-3.716922072978747, 48.692414577807426],
    [-4.61226152678168, 48.58043305574324],
    [-4.809256070643727, 48.356790369118556],
    [-4.2414718719756195, 48.31656359488068],
    [-4.303179272057463, 48.117207760505806],
    [-4.66196975346196, 48.03446778297979],
    [-4.300761468710533, 47.80150143762941],
    [-3.61099306870409, 47.79767854012465],
    [-2.4838203846507554, 47.48352353860196],
    [-2.435438788851627, 47.24722439638364],
    [-2.094670367843719, 47.189018123379896],
    [-2.0387932974817886, 46.84896044508136],
    [-1.7797314941404068, 46.428961166280175],
    [-1.0868749176955248, 46.220813092553385],
    [-1.1016449045582704, 45.83296970236847],
    [-1.2423040092627105, 45.69272085156791],
    [-1.1422602739617105, 45.18902479597256],
    [-1.2925273226015292, 44.25039579998227],
    [-1.520340244888871, 43.52619265732048],
    [-1.7602388955483548, 43.34507385031415],
    [-1.179802720769061, 43.04024724417633],
    [-0.5072276517030048, 42.81215724997199],
    [0.5028220219249135, 42.682700446534795],
    [0.683808938870726, 42.780393898722195],
    [1.2562880720953729, 42.75007381811733],
    [1.7084160545974214, 42.52983086185907],
    [2.4807210311180654, 42.387117477857714],
    [3.2497535759328002, 42.4214824532834],
    [3.0461608170357124, 42.92620809750531],
    [3.495102820187128, 43.275835924707195],
    [3.9268455731119616, 43.54613555355016],
    [4.7860214425641345, 43.368754686072265],
    [5.806407154599555, 43.14709756934133],
    [6.670123130064667, 43.13229984458391],
    [6.955546718432771, 43.501188151711034],
    [7.653409408344828, 43.6971320862304],
    [7.649404898266852, 43.9852902794824],
    [7.639366856738661, 44.167160226720654],
    [7.01413307745463, 44.25498450660078],
    [6.904981981086564, 44.57507169199434],
    [6.963365535007625, 44.872706270777826],
    [6.700068894974805, 45.09177288157247],
    [7.183118379710606, 45.35762476306937],
    [6.987247210012043, 45.648394818348436],
    [6.805459223830326, 45.833468188904476],
    [7.088650525577265, 45.897806239934795],
    [6.849313749377501, 46.20361188744724],
    [6.860419457696111, 46.38144505422733],
    [6.316858089423647, 46.32410004684533],
    [6.22389698734807, 46.19322047418092],
    [6.050013062978195, 46.15344047298771],
    [6.123134567209092, 46.32945518208527],
    [6.068073662897916, 46.52357887373702],
    [6.31366879363685, 46.66515683984167],
    [6.501808968397569, 46.91171034805248],
    [6.800593244020519, 47.139851937103884],
    [7.054321884995858, 47.30965628491924],
    [7.024251431620854, 47.503452288261514],
    [7.534968410764094, 47.51557813599004],
    [7.596446641132729, 47.97391632515013],
    [7.846641010189302, 48.48566329509575],
    [8.184673086724843, 48.890797978239675],
    [7.903001973077153, 49.050533165311435],
    [7.455449317123538, 49.17010796116196],
    [7.135245455535085, 49.17987577258128],
    [6.838326335589613, 49.188240611207675],
    [6.4906668256054445, 49.43583335747678],
    [6.035556931116304, 49.490738286385294],
    [5.647260992743412, 49.513072363248966],
    [5.441365394471205, 49.51655173008575],
    [5.220775194921686, 49.69893343174476],
    [4.837325644049429, 49.88308192519011],
    [4.8712018829007775, 50.167003256159745],
    [4.610403568076833, 49.99011637212163],
    [4.287386011071618, 49.992869118894816],
    [4.177315841394318, 50.233125473122215],
    [3.9706461751845836, 50.35425691833373],
    [3.690925415026072, 50.295395852919995],
    [3.6464999975064245, 50.50563255226845],
    [3.3664834562625856, 50.52114870775773],
    [3.2263124138072214, 50.686632272673734],
    [3.1791658412407173, 50.837287709602606],
    [2.851093519411478, 50.70115459695771],
    [2.6381822306576055, 50.82097872161975],
    [2.5160965424793744, 51.062061390974804]
 ]]
        }
    },
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
        "Allemagne": "yellow",
        "Portugal": "green",
        "Suisse": "red"
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
