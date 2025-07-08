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


rawdata = pd.read_csv("datasets/superdataset-24-f hybrid one-2Ysum.csv")

testresultsum = []
testresultextra = []

maxsaldosum = 1732
maxsaldoone = 869

signif = []
n = 50
for k in range(n):
    rawdata  = rawdata.sample(frac=1) # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop(['saldo', 'saldoone'])])
    datasetout1 = np.array(rawdata[['saldo']])
    datasetout2 = np.array(rawdata[['saldoone']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout1, test_size=0.2, random_state=42)
    trainin2, testin2, trainout2, testout2 = train_test_split(datasetin, datasetout2, test_size=0.2, random_state=42)

    # модель прогноза суммы
    modelsum = RandomForestRegressor(n_estimators=100, random_state=0)
    modelsum.fit(trainin, trainout.ravel())

    # модель одиночный прогноз
    modelone = RandomForestRegressor(n_estimators=100, random_state=0)
    modelone.fit(trainin, trainout2.ravel())

    # вычисление ошибки
    predsum = modelsum.predict(testin)
    errorsum = mean_absolute_error(testout * maxsaldosum, predsum * maxsaldosum)

    # вычисление ошибки на своём датасете
    #predtest = modelone.predict(testin)
    #testerror = mean_absolute_error(testout2 * maxsaldoone, predtest * maxsaldoone)

    # вычисление ошибки проноза с экстраполяцией
    predone = modelone.predict(testin)
    predone = predone * maxsaldoone
    predextra = predone * 2
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