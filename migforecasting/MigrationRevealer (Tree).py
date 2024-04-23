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
from sklearn.model_selection import train_test_split


# Получение данных
rawdata = pd.read_csv("datasets/citiesdataset-3.csv")

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout = np.array(rawdata[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

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

"""
features = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
            'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
            'invests', 'funds', 'companies', 'factoriescap',
            'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
            'foodservturnover']
            
            'Широта', 'Долгота', 'Доллар'
"""
features = ['Числ. насл.', 'Ср. кол-во. раб.', 'Безраб.', 'Ср. з/п', 'Площ. на чел.',
            'Дошкол.', 'Врачей на чел.', 'Коек на чел.', 'Мощ. клиник',
            'Инвест.', 'Фонды', 'Предприятия', 'Мощ. промыш.',
            'Объемы строит.', 'Постр. жил. площ.', 'Постр. кварт.', 'Оборот розницы',
            'Оборот общепит.']

important = model.feature_importances_

plt.barh(features, important)
plt.show()

print("MAPE (train): ", errortrain)
print("MAPE (test): ", errortest)
