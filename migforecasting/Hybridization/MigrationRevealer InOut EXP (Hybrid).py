import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

maxsaldoIn = 3933
maxsaldoOut = 4087

# Получение данных
rawdata = pd.read_csv("superdataset-24 InOut.csv")

rawdata = rawdata[rawdata.columns.drop('popsize')]

resulttest = []
resulttrain = []

n = 50
for k in range(n):
    rawdata = rawdata.sample(frac=1)  # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    # 1 - для модели регресси inflow, 2 - для модели ргерессии outflow
    datasetin = np.array(rawdata[rawdata.columns.drop(['inflow', 'outflow'])])
    datasetout1 = np.array(rawdata[['inflow']])
    datasetout2 = np.array(rawdata[['outflow']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout1, testout1 = train_test_split(datasetin, datasetout1, test_size=0.2, random_state=146)
    trainin, testin, trainout2, testout2 = train_test_split(datasetin, datasetout2, test_size=0.2, random_state=146)

    # модель 1
    model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    model1.fit(trainin, trainout1.ravel())
    predinflow = model1.predict(trainin)

    # модель 2
    model2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model2.fit(trainin, trainout2.ravel())
    predoutflow = model2.predict(trainin)

    #errortrain = mean_absolute_error(((trainout1 * maxsaldoIn) - (trainout2 * maxsaldoOut)),
                                     #((predinflow * maxsaldoIn) - (predoutflow * maxsaldoOut)))

    errortrain = mean_squared_error((trainout1 - trainout2), (predinflow - predoutflow))

    resulttrain.append(errortrain)

    predinflow = model1.predict(testin)
    predoutflow = model2.predict(testin)

    #errortest = mean_absolute_error(((testout1 * maxsaldoIn) - (testout2 * maxsaldoOut)),
                                    #((predinflow * maxsaldoIn) - (predoutflow * maxsaldoOut)))

    errortest = mean_squared_error((testout1 - testout2), (predinflow - predoutflow))

    resulttest.append(errortest)

    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')
