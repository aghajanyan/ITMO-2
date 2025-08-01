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


rawdata = pd.read_csv("datasets/superdataset-24-f 3Ysum.csv")

#rawdata = rawdata.sample(frac=1) # перетасовка

#rawdata = rawdata[:4135]

#rawdata = rawdata[rawdata.columns.drop('popsize')]
#rawdata = rawdata[rawdata.columns.drop('parks')]
#rawdata = rawdata[rawdata.columns.drop('museums')]

resulttest = []
resulttrain = []
#maxsaldo = 26466   # olddatasets
#maxsaldo = 39719   # dataset 00-10
#maxsaldo = 10001    # dataset 20 (also positive)
#maxsaldo = 426      # dataset 22
#maxsaldo = 848     # dataset 23
#maxsaldo = 1775     # dataset 23 (positive flow)
#maxsaldo = 888     # dataset 23 (negative flow)
#maxsaldo = 854     # dataset 24 (also balanced, normbysoul)
#maxsaldo = 1732     # 24-f 2Ysum
maxsaldo = 2483     # 24-f 3Ysum
#maxsaldo = 951     # dataset 24 balanced-f also 24-f also 2Y
#maxsaldo = 947      # 24-f 3Y
#maxsaldo = 347      # dataset 24 interreg (also balanced)
#maxsaldo = 512     # dataset 24 reg (also balanced)
#maxsaldo = 295     # dataset 24 internat
#maxsaldo = 294     # dataset 24 internat balanced
#maxsaldo = 1879     # dataset 24 (positive flow)
#maxsaldo = 888     # dataset 24 (negative flow)
#maxsaldo = 854     # dataset 24-2 (positive flow)
#maxsaldo = 1046    # dataset 24-2 (negative flow)
#maxsaldo = 2854     # negative flow (dataset 20)
#maxsaldo = 3146     # value-driven (40-series)
#maxsaldo = 4795     # value-driven (41-1 positive flow)
#maxsaldo = 3979     # value-driven negative flow (40 series & 41-1)
#maxsaldo = 23444   # value-driven (42, big cities only)
#maxsaldo = 845      # value-driven 43
#maxsaldo = 1148     # value-driven 43 (negative flow)
#maxsaldo = 1954      # value-driven 43 (positive flow)
#maxsaldo = 10624      # value-driven 44

#maxsaldo = 3933     # dataset 24 inflow
#maxsaldo = 4087     # dataset 24 outflow

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
    errortrain = mean_absolute_error(trainout * maxsaldo, predtrain * maxsaldo)

    predtest = model.predict(testin)
    errortest = mean_absolute_error(testout * maxsaldo, predtest * maxsaldo)

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