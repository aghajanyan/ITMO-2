import pandas as pd
import numpy as np
import re

df=pd.read_csv('bestcities.csv')#, encoding='cp1251', encoding_errors='strict')

names=df['0'].sort_values().unique()

coordinates=pd.DataFrame(columns=['name', 'lat', 'lon'])
coordinates.name=names

import osmnx as ox #Open street Map https://osmnx.readthedocs.io/en/stable/user-reference.html
import warnings

warnings.filterwarnings("ignore")
lat, lon = [],[]

'''
Например:
    Городской округ город Кунгур -> Кунгур
    Зеленоградский г.о. -> Зеленоградский округ
    Муниципальный район Сосногорск -> район Сосногорск
    город Апатиты -> Апатиты
    Муниципальный район Город Краснокаменск и Краснокаменский район -> Краснокаменский район
'''

for city in names: #Названия городов из таблицы с данными
    
    # убираем "городской округ", "город", "муниципальный" из названия
    # и '-курорт', тк иначе "город-курорт Железноводск" в поиске дает г.о. Железноводск, а не городок
    city = re.sub(r'([гГ]ородской округ )|([гГ]ород )|([мМ]униципальный )|(-курорт)','',city).split(' (')[0]
    # превращаем "г.о." в "округ"
    city = re.sub(r'г\.о\.','округ',city)
    # для решения проблем, когда сразу и город, и район в названии; оставляем район
    search_res = re.search(r'\w+ район', city)
    if search_res:
        city = search_res.group(0)
        
    try: 
        osm_place=ox.geocode_to_gdf(city, which_result=1)
    except:
        print(city, 'не нашлось')
        x, y = None, None
    else:
        coord=osm_place["geometry"].centroid
        x, y = (coord.y.values[0], coord.x.values[0])
        print(city, osm_place['name'][0] + ': %.4f с.ш., %.4f в.д.'%(x,y))
    finally:
        lat.append(x)
        lon.append(y)
if len(lat)==len(names):
    coordinates.lat=lat
    coordinates.lon=lon
else:
    print('')

coordinates.to_csv("coordinates.csv", index=False)
print('Done')
