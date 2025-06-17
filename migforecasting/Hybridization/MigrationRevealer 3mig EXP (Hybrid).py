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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


maxsaldoreg = 522
maxsaldointerreg = 377
maxsaldointernat = 273

# Получение данных
rawdata = pd.read_csv("superdataset-24 3mig.csv")

rawdata = rawdata.sample(frac=1) # перетасовка
rawdata = rawdata[:4758]

resulttest = []
resulttrain = []

n = 50
for k in range(n):

    # перетасовка
    rawdata = rawdata.sample(frac=1)

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    # 1 - для модели регресси reg, 2 - для модели ргерессии interreg, 3 - для модели ргерессии internat
    datasetin = np.array(rawdata[rawdata.columns.drop(['reg', 'interreg', 'internat'])])
    datasetout1 = np.array(rawdata[['reg']])
    datasetout2 = np.array(rawdata[['interreg']])
    datasetout3 = np.array(rawdata[['internat']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout1, testout1 = train_test_split(datasetin, datasetout1, test_size=0.2, random_state=146)
    trainin, testin, trainout2, testout2 = train_test_split(datasetin, datasetout2, test_size=0.2, random_state=146)
    trainin, testin, trainout3, testout3 = train_test_split(datasetin, datasetout3, test_size=0.2, random_state=146)

    # модель 1
    model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    model1.fit(trainin, trainout1.ravel())
    predreg = model1.predict(trainin)

    # модель 2
    model2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model2.fit(trainin, trainout2.ravel())
    predinterreg = model2.predict(trainin)

    # модель 3
    model3 = RandomForestRegressor(n_estimators=100, random_state=0)
    model3.fit(trainin, trainout3.ravel())
    predinternat = model3.predict(trainin)

    # ошибка гибридной модели
    errortrain = r2_score(((trainout1 * maxsaldoreg) + (trainout2 * maxsaldointerreg) + (trainout3 * maxsaldointernat)),
                                     ((predreg * maxsaldoreg) + (predinterreg * maxsaldointerreg) + (predinternat * maxsaldointernat)))


    resulttrain.append(errortrain)

    predreg = model1.predict(testin)
    predinterreg = model2.predict(testin)
    predinternat = model3.predict(testin)

    errortest = r2_score(((testout1 * maxsaldoreg) + (testout2 * maxsaldointerreg) + (testout3 * maxsaldointernat)),
                                     ((predreg * maxsaldoreg) + (predinterreg * maxsaldointerreg) + (predinternat * maxsaldointernat)))

    #errortest = mean_squared_error((testout1 + testout2 + testout3), (predreg + predinterreg + predinternat))

    resulttest.append(errortest)
    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')