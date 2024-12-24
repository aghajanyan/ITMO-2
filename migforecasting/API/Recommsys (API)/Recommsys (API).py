
"""
Recommsys (API).py ver. 0.0
"""

import pandas as pd
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi import Body
from fastapi import Request
import uvicorn

app = FastAPI()


#Нормирование цен согласно инфляции
def normbyinf(inputdata):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    infdata = pd.read_csv("inflation14.csv")
    for k in range(len(inputdata)):
        inflation = infdata[infdata['year'] == inputdata.iloc[k]['year']]   # получить инфляцию за необходимый год
        for col in thisrubfeatures:
            index = inputdata.columns.get_loc(col)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * (inflation.iloc[0]['inf'] / 100)

    return inputdata


# нормирование факторов на душу населения
def normpersoul(tonorm):
    for k in range(len(tonorm)):
        for col in features:
            index = tonorm.columns.get_loc(col)
            tonorm.iloc[k, index] = float(tonorm.iloc[k][col] / tonorm.iloc[k]['popsize'])

    return tonorm


# вычисляется во сколько раз входные данные отличаются от центра лучших кластеров
# по каждому социально-экономическому индикатору
@app.get("/recommsys")
async def reveal(request: Request):
    # обработка входных данных
    inputdata = dict(request.query_params)
    inputdata = pd.DataFrame(inputdata, index=[0])

    inputdata = normbyinf(inputdata)

    filename = ''
    # выброр медиан кластеров согласно уровню МО
    if inputdata.iloc[0]['type'] == '!mundist':
        filename = 'medians all.csv'
    else:
        filename = 'medians only mundist.csv'

    medians = pd.read_csv(filename)

    medians = normpersoul(medians)
    inputdata = normpersoul(inputdata)

    changes = []
    tmp = []
    # вычисление разницы входа от медиан лучшего кластера
    for i in range(len(medians)):
        if inputdata.iloc[0]['profile'] == medians.iloc[i]['profile']:
            for col in inputdata.iloc[:, 7:]:
                tmp.append(float(medians.iloc[i][col] / inputdata.iloc[0][col]))

            changes.append(tmp)
            tmp = []
            break

    features = list(inputdata.iloc[:, 7:].columns)
    changes = np.array(changes)
    changes = pd.DataFrame(changes, columns=features)

    return changes.to_json()

if __name__ == "__main__":
    uvicorn.run("Recommsys (API):app", host="0.0.0.0", port=8000, reload=True)