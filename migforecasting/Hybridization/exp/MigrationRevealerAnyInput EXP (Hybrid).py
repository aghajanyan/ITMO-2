import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


rawdata = pd.read_csv("citiesdataset-NYDcor-4.csv")

# Исключение из выборки отдельных признаков (отсутствуют у малых городов/райнов)
rawdata = rawdata.drop(['beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
                        'funds', 'companies', 'consnewapt', 'dollar'], axis=1)

#проверка точности предсказания на малых городах/территориях
village = pd.read_csv("input60NY.csv")

villagein = np.array(village[village.columns.drop('saldo')])
villageout = np.array(village[['saldo']])

resulttest = []
resulttrain = []
resultvillage = []
maxsaldo = 26466
for k in range(50):
    rawdata = rawdata.sample(frac=1) # перетасовка

    # создание бинарного датасета для прогнозирования оттока/притока
    rawdataclass = pd.DataFrame()
    for i in range(rawdata.shape[0]):
        if rawdata.iloc[i, rawdata.shape[1] - 1] > 0:
            rawdataclass = rawdataclass.append(rawdata.iloc[i])
            rawdataclass.iloc[i, rawdata.shape[1] - 1] = 1
        else:
            rawdataclass = rawdataclass.append(rawdata.iloc[i])
            rawdataclass.iloc[i, rawdata.shape[1] - 1] = 0

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    # 1 - для модели регресси, 2 - для модели классификатора (входные данные одинаковые)
    datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
    datasetout1 = np.array(rawdata[['saldo']])
    datasetout2 = np.array(rawdataclass[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout1, testout1 = train_test_split(datasetin, datasetout1, test_size=0.2, random_state=146)
    trainin, testin, trainout2, testout2 = train_test_split(datasetin, datasetout2, test_size=0.2, random_state=146)

    # модель 1
    model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    model1.fit(trainin, trainout1.ravel())
    predtrainreg = model1.predict(trainin)

    # модель 2
    model2 = RandomForestClassifier(n_estimators=100, random_state=0)
    model2.fit(trainin, trainout2.ravel())
    predtrainclass = model2.predict(trainin)

    # замена знака в прогнозе регрессионной модели согласно прогнозу классификатора
    for i in range(len(predtrainreg)):
        if predtrainclass[i] == 1:
            predtrainreg[i] = abs(predtrainreg[i])
        else:
            predtrainreg[i] = -abs(predtrainreg[i])

    # оценка на тестовой выборке
    predtestreg = model1.predict(testin)
    predtestclass = model2.predict(testin)

    for i in range(len(predtestreg)):
        if predtestclass[i] == 1:
            predtestreg[i] = abs(predtestreg[i])
        else:
            predtestreg[i] = -abs(predtestreg[i])

    predvillagereg = model1.predict(villagein)
    predvillageclass = model2.predict(villagein)

    for i in range(len(predvillagereg)):
        if predvillageclass[i] == 1:
            predvillagereg[i] = abs(predvillagereg[i])
        else:
            predvillagereg[i] = -abs(predvillagereg[i])

    errortrain = mean_absolute_percentage_error(trainout1, predtrainreg) #* maxsaldo
    errortest = mean_absolute_percentage_error(testout1, predtestreg) #* maxsaldo
    errorvillage = mean_absolute_percentage_error(villageout, predvillagereg) #* maxsaldo

    # запись ошибки
    resulttrain.append(errortrain)
    resulttest.append(errortest)
    resultvillage.append(errorvillage)

    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)
resultvillage = np.array(resultvillage)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)
resultvillage = pd.DataFrame(resultvillage)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')
resultvillage.to_excel('village-data.xlsx')
