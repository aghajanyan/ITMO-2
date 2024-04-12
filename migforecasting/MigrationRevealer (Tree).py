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


class Normalization:
    def normbymax(trainset):
        for k in range(len(trainset[0])):
            maxi = trainset[0][k]
            for i in range(len(trainset)):
                if (maxi < trainset[i][k]):
                    maxi = trainset[i][k]

            for j in range(len(trainset)):
                trainset[j][k] = trainset[j][k] / maxi


# Получение данных
rawdata = pd.read_csv("citiesdataset 10-21.csv")
rawdata = np.array(rawdata)

# -- Нормализация --
rawdata = np.delete(rawdata, 0, 1)  # удаляем название городов

# перевод из текста в число (удалить пример при невозможности конвертации)
i = 0
while i < len(rawdata):
    for j in range(len(rawdata[1])):
        if rawdata[i, j] == rawdata[i, j]:  # проверка NaN
            try:
                rawdata[i, j] = float(rawdata[i, j])
            except ValueError:
                rawdata = np.delete(rawdata, i, 0)
                i -= 1
                break
        else:
            rawdata = np.delete(rawdata, i, 0)
            i -= 1
            break
    i += 1

Normalization.normbymax(rawdata)

random.shuffle(rawdata)

rawdata = np.array(rawdata)

# разбиение датасета на входные признаки и выходной результат (сальдо) 
datasetin = rawdata[:, :18]
datasetout = rawdata[:, 18:]

# разбиение на обучающую и тестовую выборку
trainin, trainout, testin, testout = [], [], [], []

spliter = len(datasetin) * 0.9
trainin = np.array(datasetin[:int(spliter)])
trainout = np.array(datasetout[:int(spliter)])

testin = np.array(datasetin[int(spliter):])
testout = np.array(datasetout[int(spliter):])

# модель
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(trainin, trainout)

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
