
"""
Recommsys (API).py ver. 2.0
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

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
            infnorm = 1 - (inflation.iloc[0]['inf'] / 100)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * infnorm

    return inputdata


# Нормирование данных для модели (от 0 до 1)
def normformodel(inputdata, filename, normbysoul):
    norm = pd.read_csv(filename)
    final = []
    tmp = []
    for k in range(len(inputdata)):
        for col in norm:
            if normbysoul == True:
                if col != 'saldo' and col != 'popsize':
                    tmp.append(inputdata.iloc[k][col] / norm.iloc[0][col])
            else:
                if col != 'saldo':
                    tmp.append(inputdata.iloc[k][col] / norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final = np.array(final)
    if normbysoul == True:
        features = list(norm.columns[2:])
    else:
        features = list(norm.columns[1:])
    final = pd.DataFrame(final, columns=features)
    inputdata = final
    return inputdata


# нормирование факторов на душу населения (для старой версии планирования)
def normpersoul(tonorm, normfeat):

    for a in normfeat:
        tonorm[a] = tonorm[a] / tonorm['popsize']

    return tonorm


# обработчик входных данных
def inputproc(request):
    # обработка входных данных
    inputdata = dict(request.query_params)
    inputdata = pd.DataFrame(inputdata, index=[0])

    features = ['type', 'profile', 'year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
                'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen', 'livestock',
                'harvest', 'agrprod', 'hospitals', 'beforeschool']

    inputdata = inputdata[features]  # правильный порядок для модели
    inputdata.iloc[:, 2:] = inputdata.iloc[:, 2:].astype(float)

    return inputdata


# сохранение файла с медианами кластеров
def getmedians():
    data = pd.read_csv("superdataset-24 alltime-clust (oktmo+name+clust) 01.csv")
    norm = pd.read_csv("fornorm 24 all (IQR).csv")
    tmpdata = []
    medians = []
    tmp = []
    for k in range(int(data['clust'].max()) + 1):
        tmpdata = data[data['clust'] == k]
        for col in norm:
            tmp.append(tmpdata[col].median() * norm.iloc[0][col])

        medians.append(tmp)
        tmp = []

    features = list(norm.columns)
    medians = np.array(medians)
    medians = pd.DataFrame(medians, columns=features)

    clust = []
    for i in range(len(medians)):
        clust.append(i)

    medians['clust'] = clust
    medians.to_excel('medians 01.xlsx', index=False)


# сохранение файла с центроидами кластеров
def getcentroids():
    data = pd.read_csv("superdataset-24 alltime-clust (oktmo+name+clust) 01.csv")
    norm = pd.read_csv("fornorm 24 all (IQR).csv")
    tmpdata = []
    saldo = []
    for k in range(int(data['clust'].max()) + 1):
        tmpdata = data[data['clust'] == k]
        saldo.append(tmpdata['saldo'].median())

    # загрузка модели
    kmeans_model = joblib.load('kmeans_model (24-all-iqr) 01.joblib')

    centroids = kmeans_model.cluster_centers_
    features = list(norm.columns)
    centroids = np.array(centroids)
    centroids = pd.DataFrame(centroids, columns=features)

    clust = []
    for i in range(len(centroids)):
        clust.append(i)

    centroids['clust'] = clust
    centroids['saldo'] = saldo

    centroids.to_csv('centroids 01.csv', index=False)


# определить кластер для входных данных
@app.get("/recommsys/whatcluster")
async def whatcluster(request: Request):
    # нормализация входных данных
    inputdata = inputproc(request)
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata, 'fornorm 24 all (IQR).csv', False)

    # загрузка модели
    kmeans_model = joblib.load('kmeans_model (24-all-iqr) 01.joblib')

    pred_cluster = kmeans_model.predict(inputdata)

    # получение данных о профиле кластера
    medians = pd.read_csv('medians 01.csv')
    clust = -1
    for i in range(len(medians)):
        if medians.iloc[i]['clust'] == pred_cluster[0]:
            clust = i

    return "Муниципальное образование входит в кластер: №" + str(pred_cluster[0]) +" - " + medians.iloc[clust]['profile']


# поиск наиболее близки поселений на основе социально-экономических индикаторов
@app.get("/recommsys/siblingsfinder")
async def siblingsfinder(request: Request):

    # нормализация входных данных
    inputdata = inputproc(request)
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata, 'fornorm 24 all (IQR).csv', False)
    # факторы для нормирования
    normfeat = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool']
    inputdata = normpersoul(inputdata, normfeat)

    #загрузка датасета
    data = pd.read_csv("superdataset-24 alltime-clust (oktmo+name+clust) 01-normbysoul.csv")

    # наиболее близкие среди всех кластеров
    dist1 = []
    tmp1 = 0.0
    for b in range(len(data)):
        tmp1 = mean_squared_error(data.iloc[b][6:21], inputdata.iloc[0][1:])  # кроме popsize
        dist1.append(tmp1)

    # сортировка датафрейма согласно отклонению (dist1)
    data['dist1'] = dist1
    data = data.sort_values(by='dist1')

    # выделение топ-10 наиболее близких (похожих)
    top10 = []
    for i in range(10):
        top10.append(data.iloc[i])

    top10 = np.array(top10)
    col = list(data.columns)
    top10 = pd.DataFrame(top10, columns=col)

    return top10.iloc[:, :3].to_dict()


# разница от наиболее близкого из лучшего кластера
@app.get("/recommsys/h2h")
async def headtohead(request: Request):
    # обработка входных данных
    inputdata = inputproc(request)
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata, 'fornorm 24 all (IQR).csv', False)
    # факторы для нормирования
    normfeat = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool']
    inputdata = normpersoul(inputdata, normfeat)

    #загрузка датасета
    data = pd.read_csv("superdataset-24 alltime-clust (oktmo+name+clust) 01-normbysoul.csv")

    # наиболее близкий из лучшего кластера
    migprop = 0.0
    bestcluster = 0
    tmpdata = []
    # определение лучшего кластера
    for k in range(int(data['clust'].max()) + 1):
        tmpdata = data[data['clust'] == k]
        msaldo = tmpdata['saldo'].median()
        mpopsize = tmpdata['popsize'].median()
        if k == 0:
            migprop = float(msaldo / mpopsize)
        else:
            if migprop < float(msaldo / mpopsize):
                migprop = float(msaldo / mpopsize)
                bestcluster = k

    # вычисление наиболее близкого МО среди лучшего кластера
    dist = []
    tmpdata = data[data['clust'] == bestcluster]
    for i in range(len(tmpdata)):
        tmp = mean_squared_error(tmpdata.iloc[i][6:21], inputdata.iloc[0][1:])  # кроме popsize
        dist.append(tmp)

    # сортировка датафрейма согласно отклонению (dist)
    tmpdata['dist'] = dist
    tmpdata = tmpdata.sort_values(by='dist')

    #вычисление разницы между наиболее близким и заданным
    dif = []
    for col in inputdata:
        dif.append(float(tmpdata.iloc[0][col] / inputdata.iloc[0][col]))

    dif = np.array(dif)
    features = list(inputdata.columns)
    dif = pd.DataFrame(dif, index=features)

    return tmpdata.iloc[0].to_dict(), dif.to_dict()


# вычисляется во сколько раз входные данные отличаются от центра лучших кластеров согласно профилю
# по каждому социально-экономическому индикатору (развите на основе перехода в другой кластер!!!)
@app.get("/recommsys/plan")
async def reveal(request: Request):
    # обработка входных данных
    inputdata = inputproc(request)

    filename = ''
    # выброр медиан кластеров согласно уровню МО
    if inputdata.iloc[0]['type'] == 'all':
        filename = 'medians 01.csv'
    else:
        filename = 'medians only mundist.csv'

    medians = pd.read_csv(filename)

    # сортировка от лучшего кластера к худшему (согласно критерию)
    medians = medians.sort_values(by=['migprop'], ascending=False)

    # факторы для нормирования
    normfeat = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool']
    medians = normpersoul(medians, normfeat)
    inputdata = normbyinf(inputdata)
    inputdata = normpersoul(inputdata, normfeat)

    changes = []
    tmp = []
    # вычисление разницы входа от медиан лучшего кластера (согласно профилю)
    for i in range(len(medians)):
        if inputdata.iloc[0]['profile'] == medians.iloc[i]['profile']:
            for col in inputdata.iloc[:, 4:]:
                tmp.append(float(medians.iloc[i][col] / inputdata.iloc[0][col]))

            changes.append(tmp)
            tmp = []
            break

    features = list(inputdata.iloc[:, 4:].columns)
    changes = np.array(changes)
    changes = pd.DataFrame(changes, columns=features)

    return changes.to_json()


# вычисляется во сколько раз входные данные отличаются от медиан положительной части своего кластера
# по каждому социально-экономическому индикатору (развитие внутри кластера!!!)
@app.get("/recommsys/plan-profileless")
async def reveal(request: Request):
    # обработка входных данных
    inputdata = dict(request.query_params)
    inputdata = pd.DataFrame(inputdata, index=[0])

    features = ['year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
                'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen', 'livestock',
                'harvest', 'agrprod', 'hospitals', 'beforeschool']

    inputdata = inputdata[features]  # правильный порядок для модели
    inputdata = inputdata.astype(float)

    # нормализация инфляции, перевод в душевые показатели и нормализация под модель
    inputdata = normbyinf(inputdata)
    # факторы для нормирования
    normfeat = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']
    inputdata = normpersoul(inputdata, normfeat)
    inputdatamodel = normformodel(inputdata, 'fornorm 24 all (IQR)-normbysoul.csv', True)

    # загрузка модели и определение кластера
    kmeans_model = joblib.load('kmeans_model (24-all-iqr-normbysoul).joblib')
    pred_cluster = kmeans_model.predict(inputdatamodel)

    # загрузка медиан
    medians = pd.read_excel('medians positive.xlsx')

    changes = []
    tmp = []
    # вычисление разницы входа от медиан лучшего кластера (согласно кластеру)
    for i in range(len(medians)):
        if pred_cluster[0] == int(medians.iloc[i]['clust']):
            for col in inputdata.iloc[:, 2:]:
                tmp.append(float(medians.iloc[i][col] / inputdata.iloc[0][col]))

            changes.append(tmp)
            tmp = []
            break

    features = list(inputdata.iloc[:, 2:].columns)
    changes = np.array(changes)
    changes = pd.DataFrame(changes, columns=features)

    return changes.to_json()


if __name__ == "__main__":
    uvicorn.run("Recommsys (API):app", host="0.0.0.0", port=8000, reload=True)