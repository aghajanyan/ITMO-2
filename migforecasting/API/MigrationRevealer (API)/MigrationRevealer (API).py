import pandas as pd
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi import Body
from fastapi import Request
import uvicorn

app = FastAPI()


#Нормирование цен согласно инфляции
def normbyinf(inputdata, year):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    infdata = pd.read_csv("inflation14.csv")

    avginf = 0.0
    for i in range(len(infdata) - 1):
        avginf += np.abs(infdata.iloc[i]['inf'] - infdata.iloc[i + 1]['inf'])

    avginf = avginf / len(infdata)

    start = 2023
    if year != 2023:
        while start < year:
            infdata.loc[len(infdata)] = [start + 1] + [infdata.iloc[len(infdata) - 1] + avginf]

    for k in range(len(inputdata)):
        inflation = infdata[infdata['year'] == year]   # получить инфляцию за необходимый год
        for col in thisrubfeatures:
            index = inputdata.columns.get_loc(col)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * (inflation.iloc[0]['inf'] / 100)

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
    #inputdata = inputdata.transpose()
    features = ['year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover', 'livarea',
                'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals',
                'beforeschool']
    inputdata = inputdata[features]     # правильный порядок для модели
    inputdata = inputdata.astype(float)

    # загрузка модели
    model = joblib.load('migpred (24, tree).joblib')

    # получение периода прогноза (если 2023, то делается прогноз на один год, то есть от 2022 года)
    startyear = 2022
    endyear = inputdata.iloc[0]['year']
    inputdata = inputdata.iloc[:, 1:]  # отрезать показатель year

    dataforpred = []
    # нормализация согласно инфляции
    while startyear < endyear:
        dataforpred.append(normbyinf(inputdata, startyear + 1))
        startyear += 1

    dataforpred = np.array(dataforpred)
    dataforpred = pd.DataFrame(dataforpred, columns=inputdata.columns)

    # выполнение прогноза
    predsaldo = 0
    for i in range(len(dataforpred)):
        dataforpred.iloc[i], maxsaldo = normformodel(dataforpred.iloc[[i]])

        prediction = model.predict(dataforpred)
        prediction = prediction * maxsaldo
        predsaldo += int(prediction)

    return predsaldo


@app.get("/notsure")
async def calc(year: int, popsize: int, avgemployers: float, avgsalary: float, shoparea: float,
               foodseats: int, retailturnover: float, livarea: int, sportsvenue: int, servicesnum: int, roadslen: float,
               livestock: int, harvest: float, agrprod: float, hospitals: int, beforeschool: int, factoriescap: float):
    return avgsalary


if __name__ == "__main__":
    uvicorn.run("MigrationRevealer (API):app", host="0.0.0.0", port=8000, reload=True)
