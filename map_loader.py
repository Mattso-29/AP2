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
# Coordonnées simplifiées des polygones pour chaque pays
geojson_data = {
    "France 🇫🇷": {
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
    "Germany 🇩🇪": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [8.190095291299144, 48.868005503091126],
                [7.62241737678076, 48.003042209435705],
                [7.5656229637448575, 47.553653216583],
                [7.896808755147617, 47.55614460467925],
                [8.559063030737093, 47.64335481569188],
                [8.416933022728017, 47.71823987797879],
                [8.591017012123672, 47.82501987901202],
                [8.780461773929176, 47.71774563323319],
                [9.15913354690116, 47.65164029189188],
                [9.59855124947407, 47.486914489073285],
                [9.821181469058928, 47.56977562502962],
                [10.055933134577202, 47.48048149247572],
                [10.224436364863436, 47.2950710245494],
                [10.448957219907669, 47.41956618197264],
                [10.437609572396696, 47.559259366646586],
                [10.831737658187706, 47.50756869258964],
                [11.080656042320868, 47.393742445694926],
                [11.70869572436439, 47.568259272235366],
                [12.142256852133585, 47.60671428497034],
                [12.26220524851459, 47.7312086091286],
                [12.722849792782, 47.647905567825916],
                [13.016078336415006, 47.47293578041092],
                [13.1246883137174, 47.62974730272518],
                [12.967829519174956, 47.69092984119041],
                [12.96722629649912, 47.895551880301724],
                [12.738697127593298, 48.08812365006165],
                [12.963510542422398, 48.261064175685306],
                [13.41335547289799, 48.358879326137156],
                [13.46264840974112, 48.539153313561485],
                [13.721093774317637, 48.48374784015266],
                [13.859713976896984, 48.713332437574934],
                [13.753892896537508, 48.81518860642345],
                [13.60040352429084, 48.940502028856486],
                [13.250098884568644, 49.0954200580199],
                [12.969995565348398, 49.299530246733895],
                [12.621375576073177, 49.494318744381985],
                [12.456674460596929, 49.713576652135515],
                [12.538793422374255, 49.89229517437167],
                [12.253111489498366, 50.083053526320896],
                [12.149706638448407, 50.25698012691146],
                [12.359008312898453, 50.143790369933186],
                [12.564850841479483, 50.37224590086001],
                [12.958983986519456, 50.41229058878858],
                [13.337701441265835, 50.602026535907754],
                [14.377158538480558, 50.89264879021616],
                [14.308385640156928, 51.025231595244406],
                [14.66289108932395, 50.93361315682495],
                [14.789356426887366, 50.84115635496855],
                [15.050574374675506, 51.27271901775117],
                [14.68605599342601, 51.55005905774448],
                [14.689592629139298, 51.90889885185575],
                [14.561491482140923, 52.44964500202309],
                [14.152846231605537, 52.84933971878476],
                [14.40910916076109, 53.21797181848527],
                [14.269212851456587, 53.62372984000356],
                [14.17959269534876, 53.92077807769283],
                [13.827056439915623, 54.162926785304876],
                [13.630334378132233, 54.38438279978379],
                [13.641866195306847, 54.610769044929754],
                [13.305206435764006, 54.69062831919058],
                [13.085278120358169, 54.44547634032784],
                [12.782246265122808, 54.41636217088066],
                [12.49149738252703, 54.47929638614988],
                [12.284108494330468, 54.27339235956481],
                [11.769981670247347, 54.14283974212623],
                [11.571933263209417, 53.96303909816325],
                [11.35825370407187, 53.83650355249654],
                [11.150594420726492, 54.039731152389294],
                [10.695648228068933, 53.97045627440082],
                [11.124049482012111, 54.265156513242175],
                [11.111302708477297, 54.384600732278585],
                [11.23163584578245, 54.461332427166724],
                [10.509916325065447, 54.36200209747227],
                [10.108467867448208, 54.44515860501183],
                [9.991758004959506, 54.44988528707593],
                [9.983957870893562, 54.79375303975563],
                [9.526075117723055, 54.83222093539688],
                [8.613477314581537, 54.88165828986581],
                [8.859919735439334, 54.49600640602505],
                [8.588242633144054, 54.32961089711101],
                [8.939059321356098, 54.058044012523695],
                [8.952100415801826, 53.83935789690281],
                [8.617354770322642, 53.88020220149887],
                [8.471380153518895, 53.60026001858486],
                [8.230186580417747, 53.5112006819692],
                [7.939588595529784, 53.70640907464418],
                [7.27065192194207, 53.68628980347242],
                [7.0408123181032325, 53.38444417899712],
                [7.179004772845536, 52.993415783976076],
                [7.0454195467967935, 52.6254670478953],
                [6.73377967916951, 52.60669314579096],
                [6.986569701850995, 52.45595325297762],
                [7.032017028815014, 52.215769897091164],
                [6.725583145646474, 52.03561319739753],
                [6.512903488944202, 51.836088408982974],
                [5.979758760958109, 51.8190022335273],
                [6.182920067826181, 51.507692838811764],
                [6.115962637783024, 51.16932345843754],
                [5.974361832434601, 51.00621605703644],
                [6.102895381463894, 50.68785131762226],
                [6.383256761534414, 50.322588833472125],
                [6.146969160220055, 50.06096434320812],
                [6.505938688531831, 49.760829272580395],
                [6.498234914664399, 49.45155511640503],
                [6.865299630367751, 49.17883230472116],
                [7.303989859708448, 49.224919172151104],
                [7.84972554812745, 49.07511032916185],
                [8.190095291299144, 48.868005503091126]
            ]]
        }
    },
    "Portugal 🇵🇹": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-7.887348825455604, 41.92036396781188],
                [-8.117758888757393, 41.811220523726284],
                [-8.175361404582503, 41.89516021461539],
                [-8.088957630844206, 42.0503726111968],
                [-8.194562243191342, 42.14853999461434],
                [-8.590579539492026, 42.065960762438934],
                [-8.871391804139904, 41.88353595768686],
                [-8.883392328269721, 41.75845426524599],
                [-8.780187820749404, 41.38955858905433],
                [-8.636181531185997, 41.02843482523923],
                [-8.77058740144497, 40.61432473911961],
                [-8.895392852399624, 40.18296441199129],
                [-9.077800819180624, 39.608455864264386],
                [-9.375413817611843, 39.334250523773306],
                [-9.413815494829407, 39.04405456036014],
                [-9.471418010654531, 38.72270754701458],
                [-9.260208785961623, 38.66276179133439],
                [-9.193005850832122, 38.42247678770855],
                [-8.924194110312811, 38.48262328774237],
                [-8.77058740144497, 38.27189129282516],
                [-8.81858949796569, 37.98491715792561],
                [-8.895392852399624, 37.94707345831168],
                [-8.77058740144497, 37.886482978231314],
                [-8.77058740144497, 37.74997200121686],
                [-8.799388659358158, 37.575174195062246],
                [-8.799388659358158, 37.48381203237125],
                [-8.847390755878905, 37.33892654755063],
                [-8.876192013792092, 37.21670049875907],
                [-8.962595787530404, 37.05597783213257],
                [-8.799388659358158, 37.048315858012984],
                [-8.54017733814328, 37.124900780532286],
                [-8.328968113450344, 37.08661799210637],
                [-8.117758888757407, 37.06363903266089],
                [-7.935350921976408, 36.97165358225742],
                [-7.820145890324909, 37.009994385600564],
                [-7.647338342849594, 37.09427609767958],
                [-7.426528698852223, 37.15551307747759],
                [-7.416928279547818, 37.22434544035836],
                [-7.445729537459755, 37.39233801148501],
                [-7.445729537459755, 37.49904683000432],
                [-7.522532891893661, 37.54473257542469],
                [-7.464930376068537, 37.674023532021536],
                [-7.368926183025877, 37.79550370864612],
                [-7.244120732071224, 37.94707345831168],
                [-7.100114442507845, 38.02274134944915],
                [-6.994509830160723, 38.037865562353744],
                [-6.946507733640004, 38.15874681310254],
                [-6.936907314335599, 38.226654612004836],
                [-7.09051402320344, 38.17384290681207],
                [-7.13851611972413, 38.25681552764371],
                [-7.311323667200725, 38.414954950252366],
                [-7.234520312766847, 38.61776951510825],
                [-7.253721151375601, 38.73019723406796],
                [-7.080913603899006, 38.82001222541223],
                [-6.936907314335599, 39.00676332674098],
                [-7.013710668769477, 39.118577997379134],
                [-7.109714861812222, 39.09622923231163],
                [-7.109714861812222, 39.15581019004932],
                [-7.244120732071224, 39.18558176692895],
                [-7.224919893462442, 39.282252325060455],
                [-7.3209240865051015, 39.35652363731214],
                [-7.263321570680034, 39.46778270024933],
                [-7.474530795372999, 39.54925980795079],
                [-7.551334149806877, 39.64542772196856],
                [-7.311323667200725, 39.630641349544334],
                [-7.013710668769477, 39.66021093315615],
                [-6.946507733640004, 39.80786910965995],
                [-6.88890521781488, 39.918405083449954],
                [-6.86970437920607, 40.014058843040715],
                [-7.008835455842589, 40.13772703929165],
                [-6.955958146393732, 40.26139523554259],
                [-6.773400173060832, 40.33279425506406],
                [-6.792300998565842, 40.6515296670874],
                [-6.887705165402123, 41.041013968706274],
                [-6.675295888295178, 41.22508029992569],
                [-6.405284095363166, 41.33799749912714],
                [-6.173673979648719, 41.551800491550985],
                [-6.345281474712015, 41.687108964735074],
                [-6.536089808384066, 41.72200107275961],
                [-6.591292219383675, 41.96383175003449],
                [-7.1697174824636924, 41.97792740620342],
                [-7.364125973375515, 41.84918656844773],
                [-7.887348825455604, 41.92036396781188]
            ]]
        }
    },
    "Switzerland 🇨🇭": {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [7.169835086839441, 45.88544370119848],
                [7.095586577067918, 45.875083121436475],
                [6.992385503468452, 46.019180910926224],
                [6.853539338534683, 46.206527419728836],
                [6.846070937722789, 46.38724064302782],
                [6.346408951844905, 46.328050158904375],
                [6.231689923433758, 46.191603021664136],
                [6.053694726613083, 46.17664483259057],
                [6.1335870838630635, 46.32934473090795],
                [6.116970895022611, 46.415374094331206],
                [6.078137216731081, 46.51469440781044],
                [6.328829176713807, 46.66712858892248],
                [6.569640022232832, 46.95861136435961],
                [7.022309149443053, 47.28341473441237],
                [7.058269594373103, 47.34649209197589],
                [7.045975766256543, 47.5010939907212],
                [7.436374858216055, 47.49684358028614],
                [7.919117658676711, 47.549151232457625],
                [8.172552627168706, 47.59158870218113],
                [8.421162168356062, 47.607980703475576],
                [8.558246079108017, 47.64817412682084],
                [8.44923319732638, 47.714402978817276],
                [8.597939908911915, 47.81438693297015],
                [8.808495640599155, 47.696232223420154],
                [9.162051121710917, 47.65447522776724],
                [9.48182861169007, 47.5474740830368],
                [9.61276479285176, 47.47768200091822],
                [9.65684328253218, 47.40788991879964],
                [9.667840328902598, 47.37521637759963],
                [9.555670455922037, 47.30678803542339],
                [9.472092903505136, 47.18181595784263],
                [9.524878726085394, 47.11959328674191],
                [9.480890540602374, 47.05737061564119],
                [9.787708134343694, 47.03319253183095],
                [9.884207216246477, 47.02412726707381],
                [9.883932290086037, 46.92505624909002],
                [10.068132816795039, 46.88900340774007],
                [10.10524784829538, 46.85210798377227],
                [10.168755791085694, 46.857330819585044],
                [10.24298585408876, 46.921882707648166],
                [10.409041254285803, 46.99681657817136],
                [10.48657043119799, 46.940207799933496],
                [10.467325600049284, 46.85355667319274],
                [10.390896127773427, 46.67348207313643],
                [10.489044766631466, 46.61061652880427],
                [10.455228849041848, 46.535660225497246],
                [10.108821888367006, 46.61661496569012],
                [10.041190053187208, 46.490634599587985],
                [10.158308597033738, 46.410165627173],
                [10.140713322841634, 46.300872049656164],
                [10.17150505267881, 46.24310026017787],
                [10.079129863166173, 46.22180056543593],
                [10.021945222037914, 46.27959479114628],
                [9.98675467365237, 46.361618673354656],
                [9.88558184704192, 46.370724847752314],
                [9.744819653497643, 46.346438342685076],
                [9.718426742208806, 46.291754236695255],
                [9.58542130068841, 46.29058490512185],
                [9.4669995122959, 46.37235337807451],
                [9.453335459788804, 46.51045232808329],
                [9.416897986437732, 46.466549721856694],
                [9.371351144747365, 46.507317602768126],
                [9.284812145536904, 46.48850545435542],
                [9.257484040523906, 46.44144660821985],
                [9.289366829705926, 46.40062936825518],
                [9.284812145536904, 46.35663816570974],
                [9.284812145536904, 46.29058490512185],
                [9.20282783049663, 46.1992373783755],
                [9.098070094610051, 46.12036741937587],
                [9.098070094610051, 46.07615085822019],
                [9.029749832075709, 46.04454589143285],
                [9.00697641123051, 45.97495128133343],
                [9.025195147906686, 45.936953675520954],
                [9.088960726272006, 45.89576019997904],
                [9.029749832075709, 45.81963036518471],
                [8.920437412020078, 45.835499342834254],
                [8.94776551703427, 45.870395169411665],
                [8.897663991176103, 45.95595573525324],
                [8.788351571120444, 45.984446611799655],
                [8.829343728641703, 46.0919465575233],
                [8.715476624417022, 46.0919465575233],
                [8.606164204362614, 46.155084129680944],
                [8.460414310955855, 46.24020507964099],
                [8.437640890110686, 46.29687912789802],
                [8.47407836346295, 46.384922293059105],
                [8.437640890110686, 46.463412465457026],
                [8.305555049211051, 46.4288907120785],
                [8.319219101718147, 46.40062936825518],
                [8.273672260027809, 46.36292479314707],
                [8.21446136583154, 46.30946540325735],
                [8.137031734959123, 46.30317262727428],
                [8.086930209100956, 46.26854942852262],
                [8.118812998283005, 46.246505089875996],
                [8.164359839973343, 46.18977895248844],
                [8.15525047163527, 46.14246240388232],
                [8.027719314904658, 46.09826357091029],
                [8.018609946566613, 46.016085953093665],
                [7.8956334740038585, 45.984446611799655],
                [7.831867895638538, 45.924282016933006],
                [7.603935460709835, 45.99015578378862],
                [7.3929619873358945, 45.92071933114531],
                [7.169835086839441, 45.88544370119848]
            ]]
        }
    }
}