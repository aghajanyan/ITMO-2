import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


rawdatasum = pd.read_csv("datasets/superdataset-24-f 3Ysum.csv")
rawdataone = pd.read_csv("datasets/superdataset-24-f 3Y.csv")

testresultsum = []
testresultextra = []

maxsaldosum = 2483     # 24-f 3Ysum
#maxsaldoone = 951     # dataset 24 balanced-f also 24-f also 2Y
maxsaldoone = 947          # 24-f 3Y

signif = []
n = 50
for k in range(n):
    rawdatasum  = rawdatasum.sample(frac=1) # перетасовка
    rawdataone  = rawdataone.sample(frac=1)  # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdatasum[rawdatasum.columns.drop('saldo')])
    datasetout = np.array(rawdatasum[['saldo']])

    datasetin2 = np.array(rawdataone[rawdataone.columns.drop('saldo')])
    datasetout2 = np.array(rawdataone[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)
    trainin2, testin2, trainout2, testout2 = train_test_split(datasetin2, datasetout2, test_size=0.2, random_state=42)

    # модель прогноза суммы
    modelsum = RandomForestRegressor(n_estimators=100, random_state=0)
    modelsum.fit(trainin, trainout.ravel())

    # модель одиночный прогноз
    modelone = RandomForestRegressor(n_estimators=100, random_state=0)
    modelone.fit(trainin2, trainout2.ravel())

    # вычисление ошибки
    predsum = modelsum.predict(testin)
    errorsum = r2_score(testout * maxsaldosum, predsum * maxsaldosum)

    # вычисление ошибки на своём датасете
    predtest = modelone.predict(testin2)
    testerror = r2_score(testout2 * maxsaldoone, predtest * maxsaldoone)

    # перенормализация тестовой выборки под другую модель
    normsum = pd.read_csv("datasets/fornorm 24-f 3Ysum.csv")
    normone = pd.read_csv("datasets/fornorm 24-f 3Y.csv")

    testin = pd.DataFrame(data=testin,columns=normsum.columns[1:])

    for a in testin.columns:
        testin[a] = testin[a] * normsum.iloc[0][a]

    for a in testin.columns:
        testin[a] = testin[a] / normone.iloc[0][a]

    testin = np.array(testin)

    # вычисление ошибки проноза с экстраполяцией
    predone = modelone.predict(testin)
    predone = predone * maxsaldoone
    predextra = predone * 3
    errorextra = r2_score(testout * maxsaldosum, predextra)

    # запись ошибки
    testresultsum.append(errorsum)
    testresultextra.append(errorextra)

    print('Итерация: ' + str(k))


testresultsum = np.array(testresultsum)
testresultextra = np.array(testresultextra)

testresultsum = pd.DataFrame(testresultsum)
testresultextra = pd.DataFrame(testresultextra)

testresultsum.to_excel('test-sum.xlsx')
testresultextra.to_excel('test-extra.xlsx')