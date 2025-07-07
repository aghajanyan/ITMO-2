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
rawdataone = pd.read_csv("datasets/superdataset-24-f.csv")
rawdatatwo = pd.read_csv("datasets/superdataset-24-f 2Y.csv")
rawdatathree = pd.read_csv("datasets/superdataset-24-f 3Y.csv")

testresultsum = []
testresultextra = []

maxsaldosum = 2483     # 24-f 3Ysum
#maxsaldosum = 1732     # 24-f 2Ysum
maxsaldoone = 951     # dataset 24 balanced-f also 24-f also 2Y
maxsaldotwo = 951     # dataset 24 balanced-f also 24-f also 2Y
maxsaldothree = 947          # 24-f 3Y

signif = []
n = 50
for k in range(n):
    rawdatasum  = rawdatasum.sample(frac=1) # перетасовка
    rawdataone  = rawdataone.sample(frac=1)  # перетасовка
    rawdatatwo = rawdatatwo.sample(frac=1)  # перетасовка
    rawdatathree = rawdatatwo.sample(frac=1)  # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdatasum[rawdatasum.columns.drop('saldo')])
    datasetout = np.array(rawdatasum[['saldo']])

    datasetin2 = np.array(rawdataone[rawdataone.columns.drop('saldo')])
    datasetout2 = np.array(rawdataone[['saldo']])

    datasetin3 = np.array(rawdatatwo[rawdatatwo.columns.drop('saldo')])
    datasetout3 = np.array(rawdatatwo[['saldo']])

    datasetin4 = np.array(rawdatathree[rawdatathree.columns.drop('saldo')])
    datasetout4 = np.array(rawdatathree[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)
    trainin2, testin2, trainout2, testout2 = train_test_split(datasetin2, datasetout2, test_size=0.2, random_state=42)
    trainin3, testin3, trainout3, testout3 = train_test_split(datasetin3, datasetout3, test_size=0.2, random_state=42)
    trainin4, testin4, trainout4, testout4 = train_test_split(datasetin4, datasetout4, test_size=0.2, random_state=42)

    # модель прогноза суммы
    modelsum = RandomForestRegressor(n_estimators=100, random_state=0)
    modelsum.fit(trainin, trainout.ravel())

    # модель одиночный прогноз
    modelone = RandomForestRegressor(n_estimators=100, random_state=0)
    modelone.fit(trainin2, trainout2.ravel())

    # модель одиночный прогноз
    modeltwo = RandomForestRegressor(n_estimators=100, random_state=0)
    modeltwo.fit(trainin3, trainout3.ravel())

    # модель одиночный прогноз
    modelthree = RandomForestRegressor(n_estimators=100, random_state=0)
    modelthree.fit(trainin4, trainout4.ravel())

    # вычисление ошибки
    predsum = modelsum.predict(testin)
    errorsum = r2_score(testout * maxsaldosum, predsum * maxsaldosum)

    # вычисление ошибки на своём датасете
    predtest = modelone.predict(testin2)
    testerror = r2_score(testout2 * maxsaldoone, predtest * maxsaldoone)

    predtest2 = modeltwo.predict(testin3)
    testerror2 = r2_score(testout3 * maxsaldotwo, predtest2 * maxsaldotwo)

    predtest3 = modelthree.predict(testin4)
    testerror3 = r2_score(testout4 * maxsaldothree, predtest3 * maxsaldothree)

    # перенормализация тестовой выборки под другую модель
    normsum = pd.read_csv("datasets/fornorm 24-f 3Ysum.csv")
    normone = pd.read_csv("datasets/fornorm 24-f.csv")
    normtwo = pd.read_csv("datasets/fornorm 24-f 2Y.csv")
    normthree = pd.read_csv("datasets/fornorm 24-f 3Y.csv")

    testinone = pd.DataFrame(data=testin,columns=normsum.columns[1:])
    testintwo = pd.DataFrame(data=testin, columns=normsum.columns[1:])
    testinthree = pd.DataFrame(data=testin, columns=normsum.columns[1:])

    for a in testinone.columns:
        testinone[a] = testinone[a] * normsum.iloc[0][a]
        testintwo[a] = testintwo[a] * normsum.iloc[0][a]
        testinthree[a] = testinthree[a] * normsum.iloc[0][a]

    for a in testinone.columns:
        testinone[a] = testinone[a] / normone.iloc[0][a]
        testintwo[a] = testintwo[a] / normtwo.iloc[0][a]
        testinthree[a] = testinthree[a] / normthree.iloc[0][a]

    testinone = np.array(testinone)
    testintwo = np.array(testintwo)
    testinthree = np.array(testinthree)

    # вычисление ошибки проноза с экстраполяцией
    predone = modelone.predict(testin)
    predtwo = modeltwo.predict(testin)
    predthree = modelthree.predict(testin)
    predone = predone * maxsaldoone
    predtwo = predtwo * maxsaldotwo
    predthree = predthree * maxsaldothree
    predextra = predone + predtwo + predthree
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