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


rawdata = pd.read_csv("datasets/citiesdataset-NYDcor-4.csv")

resulttest = []
resulttrain = []
maxsaldo = 26466
for k in range(50):
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
    errortrain = mean_squared_error(trainout, predtrain) #* maxsaldo

    predtest = model.predict(testin)
    errortest = mean_squared_error(testout, predtest) #* maxsaldo

    # запись ошибки
    resulttrain.append(errortrain)
    resulttest.append(errortest)

    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')