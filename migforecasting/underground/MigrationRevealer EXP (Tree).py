import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


rawdata = pd.read_csv("datasets/superdataset-23.csv")

rawdata = rawdata[rawdata.columns.drop('consnewareas')]
#rawdata = rawdata[rawdata.columns.drop('parks')]
#rawdata = rawdata[rawdata.columns.drop('museums')]

resulttest = []
resulttrain = []
#maxsaldo = 26466   # olddatasets
#maxsaldo = 39719   # dataset 00-10
#maxsaldo = 10001    # dataset 20 (also positive)
#maxsaldo = 426      # dataset 22
maxsaldo = 848     # dataset 23
#maxsaldo = 1775     # dataset 23 (positive flow)
#maxsaldo = 888     # dataset 23 (negative flow)
#maxsaldo = 2854     # negative flow (dataset 20)
#maxsaldo = 3146     # value-driven (40-series)
#maxsaldo = 4795     # value-driven (41-1 positive flow)
#maxsaldo = 3979     # value-driven negative flow (40 series & 41-1)
#maxsaldo = 23444   # value-driven (42, big cities only)

signif = []
n = 50
for k in range(n):
    rawdata = rawdata.sample(frac=1) # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
    datasetout = np.array(rawdata[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

    # модель
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(trainin, trainout.ravel())

    # вычисление ошибки
    predtrain = model.predict(trainin)
    errortrain = mean_absolute_error(trainout, predtrain) * maxsaldo

    predtest = model.predict(testin)
    errortest = mean_absolute_error(testout, predtest) * maxsaldo

    # запись ошибки
    resulttrain.append(errortrain)
    resulttest.append(errortest)

    print('Итерация: ' + str(k))

    # вычисление средней значимости признаков
    important = model.feature_importances_
    for i, v in enumerate(important):
        if k == 0:
            signif.append(v)
        else:
            signif[i]+= v

for i in range(len(signif)):
    signif[i] = signif[i] / n

signif = np.array(signif)
signif = pd.DataFrame(signif)
signif.to_excel('feature significance.xlsx')

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')