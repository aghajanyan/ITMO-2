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


rawdata = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/datasets/citiesdataset-NYOCor-4.csv")

# Исключение из выборки отдельных признаков (отсутствуют у малых городов/райнов)
rawdata = rawdata.drop(['beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
                        'funds', 'companies', 'consnewapt'], axis=1)

#проверка точности предсказания на малых городах/территориях
village = pd.read_csv("inputNYO.csv")

villagein = np.array(village[village.columns.drop('saldo')])
villageout = np.array(village[['saldo']])

resulttest = []
resulttrain = []
resultvillage = []
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

    predtrain = model.predict(trainin)
    errortrain = mean_absolute_percentage_error(trainout * maxsaldo, predtrain * maxsaldo)

    predtest = model.predict(testin)
    errortest = mean_absolute_percentage_error(testout * maxsaldo, predtest * maxsaldo)

    predvillage = model.predict(villagein)
    errorvillage = mean_absolute_percentage_error(villageout * maxsaldo, predvillage * maxsaldo)

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

resulttest.to_excel('testMAPE.xlsx')
resulttrain.to_excel('trainMAPE.xlsx')
resultvillage.to_excel('villageMAPE.xlsx')