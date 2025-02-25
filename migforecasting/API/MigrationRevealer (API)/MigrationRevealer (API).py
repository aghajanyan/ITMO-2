# version 0.8 (25.02.2025)

import pandas as pd
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi import Body
from fastapi import Request
import uvicorn

app = FastAPI()


#Нормирование цен согласно инфляции
def normbyinf(inputdata, infdata, year):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']

    for k in range(len(inputdata)):
        inflation = infdata[infdata['year'] == year]   # получить инфляцию за необходимый год
        for col in thisrubfeatures:
            index = inputdata.columns.get_loc(col)
            infnorm = 1 - (inflation.iloc[0]['inf'] / 100)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * infnorm

    return inputdata.iloc[0]


# Нормирование данных для модели (от 0 до 1)
def normformodel(inputdata):
    norm = pd.read_csv("fornorm-24.csv")
    final = []
    tmp = []
    for k in range(len(inputdata)):
        for col in norm:
            if col != 'saldo':
                tmp.append(inputdata.iloc[k][col] / norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final = np.array(final)
    features = list(norm.columns[1:])
    final = pd.DataFrame(final, columns=features)
    inputdata = final
    return inputdata.iloc[0], norm.iloc[0]['saldo']


@app.get("/migration-revealer")
async def reveal(request: Request):
    # обработка входных параметров
    inputdata = dict(request.query_params)
    inputdata = pd.DataFrame(inputdata, index=[0])
    features = ['year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover', 'livarea',
                'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals',
                'beforeschool']
    inputdata = inputdata[features]     # правильный порядок для модели
    inputdata = inputdata.astype(float)

    # загрузка модели
    model = joblib.load('migpred (24, tree).joblib')

    startyear = 2024    # начальная точка прогноза, т.е. первый прогноз делается на 25-ый год
    endyear = int(inputdata.iloc[0]['year'])
    inputdata = inputdata.iloc[:, 1:]  # отрезать показатель year

    # нормализация согласно инфляции
    infdata = pd.read_csv("inflation14.csv")
    dataforpred = []
    dataforpred.append(np.array(normbyinf(inputdata, infdata, startyear)))

    # список в датафрейм
    dataforpred = pd.DataFrame(dataforpred, columns=inputdata.columns)

    #нормализация под модель прогноза
    maxsaldo = 0
    for i in range(len(dataforpred)):
        dataforpred.iloc[i], maxsaldo = normformodel(dataforpred.iloc[[i]])

    # выполнение прогноза
    predsaldo = 0
    prediction = model.predict(dataforpred)
    prediction = prediction * maxsaldo
    predsaldo = int(np.sum(prediction))

    return 'Миграционное сальдо к ' + str(endyear) + ': ' + str(predsaldo * (endyear - startyear))


if __name__ == "__main__":
    uvicorn.run("MigrationRevealer (API):app", host="0.0.0.0", port=8000, reload=True)
