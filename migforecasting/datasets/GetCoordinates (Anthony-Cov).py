import pandas as pd
import numpy as np

df=pd.read_csv('citiesdataset 10-21 (FD+Inv).csv')#, encoding='cp1251', encoding_errors='strict')

names=df.name.sort_values().unique()

coordinates=pd.DataFrame(columns=['name', 'lat', 'lon'])
coordinates.name=names

import osmnx as ox #Open street Map https://osmnx.readthedocs.io/en/stable/user-reference.html
import warnings
warnings.filterwarnings("ignore")
lat, lon = [],[]
for city in names: #Названия городов из таблицы с данными
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
