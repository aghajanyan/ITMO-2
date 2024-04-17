# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:09:34 2024

@author: Albert 
"""

import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Получение данных
rawdata = pd.read_csv("datasets/citiesdataset-NY-1.csv")
rawdata = np.array(rawdata)

rawdata = rawdata.sample(frac=1)    # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout = np.array(rawdata[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin, trainout, testin, testout = [], [], [], []

spliter = len(datasetin) * 0.9
trainin = np.array(datasetin[:int(spliter)])
trainout = np.array(datasetout[:int(spliter)])

testin = np.array(datasetin[int(spliter):])
testout = np.array(datasetout[int(spliter):])

# модель
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(trainin, trainout.ravel())

predtrain = model.predict(trainin)
errortrain = mean_absolute_percentage_error(trainout, predtrain)

predtest = model.predict(testin)
errortest = mean_absolute_percentage_error(testout, predtest)

# вывод результатов
plt.plot(predtest[:100], label='Предсказание')
plt.plot(testout[:100], label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Миграционное сальдо")
plt.title("Прогноз на тестовой выборке")
plt.show()

features = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
            'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
            'invests', 'funds', 'companies', 'factoriescap',
            'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
            'foodservturnover']

important = model.feature_importances_

plt.barh(features, important)
plt.show()

print("MAPE (train): ", errortrain)
print("MAPE (test): ", errortest)
