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


rawdata = pd.read_csv("datasets/superdataset-24-f hybrid onetwo-2Ysum.csv")

testresultsum = []
testresultextra = []

maxsaldosum = 1501
maxsaldoone = 848
maxsaldotwo = 848

signif = []
n = 50
for k in range(n):
    rawdata  = rawdata.sample(frac=1) # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop(['saldo', 'saldoone', 'saldotwo'])])
    datasetout1 = np.array(rawdata[['saldo']])
    datasetout2 = np.array(rawdata[['saldoone']])
    datasetout3 = np.array(rawdata[['saldotwo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout1, test_size=0.2, random_state=42)
    trainin2, testin2, trainout2, testout2 = train_test_split(datasetin, datasetout2, test_size=0.2, random_state=42)
    trainin3, testin3, trainout3, testout3 = train_test_split(datasetin, datasetout3, test_size=0.2, random_state=42)

    # модель прогноза суммы
    modelsum = RandomForestRegressor(n_estimators=100, random_state=0)
    modelsum.fit(trainin, trainout.ravel())

    # модель одиночный прогноз
    modelone = RandomForestRegressor(n_estimators=100, random_state=0)
    modelone.fit(trainin, trainout2.ravel())

    # модель одиночный прогноз
    modeltwo = RandomForestRegressor(n_estimators=100, random_state=0)
    modeltwo.fit(trainin, trainout3.ravel())

    # вычисление ошибки
    predsum = modelsum.predict(testin)
    errorsum = mean_absolute_error(testout * maxsaldosum, predsum * maxsaldosum)

    # вычисление ошибки на своём датасете
    #predtest = modelone.predict(testin)
    #testerror = r2_score(testout2 * maxsaldoone, predtest * maxsaldoone)

    # вычисление ошибки на своём датасете
    #predtest2 = modeltwo.predict(testin)
    #testerror2 = r2_score(testout3 * maxsaldotwo, predtest2 * maxsaldotwo)

    # вычисление ошибки проноза с экстраполяцией
    predone = modelone.predict(testin)
    predtwo = modeltwo.predict(testin)
    predone = predone * maxsaldoone
    predtwo = predtwo * maxsaldotwo
    predextra = predone + predtwo
    errorextra = mean_absolute_error(testout * maxsaldosum, predextra)

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