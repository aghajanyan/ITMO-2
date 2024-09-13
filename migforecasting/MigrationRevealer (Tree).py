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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

#Наименьшие квадраты для одной переменной
def MLS(x, y):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    n = len(x)
    sumx, sumy = sum(x), sum(y)
    sumx2 = sum([t * t for t in x])
    sumxy = sum([t * u for t, u in zip(x, y)])
    a = (n * sumxy - (sumx * sumy)) / (n * sumx2 - sumx * sumx)
    b = (sumy - a * sumx) / n
    return a, b

# Разбиение прогноза по вероятным направлениям мигрирования
def migprop(model, data, maxsaldo):
    proprates = pd.read_csv("mig whereabouts/migprop (avgalltime).csv")
    pred = model.predict(data)
    pred = pred * maxsaldo

    pred = pred.reshape(-1, 1)
    prop = [['regional', 'national', 'international']]
    for i in range(len(pred)):
        prop.append([pred[i] * proprates.iloc[0, 2],
                     pred[i] * proprates.iloc[0, 3],
                     pred[i] * proprates.iloc[0, 4]])

    return prop

#maxsaldo = 26466
#maxsaldo = 39719
#maxsaldo = 10001    # dataset 20 (also positive flow)
#maxsaldo = 426      # dataset 22
#maxsaldo = 854     # dataset 24 (also balanced)
#maxsaldo = 1080      # dataset 25, 28
#maxsaldo = 1277     # dataset 26
#maxsaldo = 951     # dataset 27
#maxsaldo = 2854     # negative flow (dataset 20)
#maxsaldo = 3146     # value-driven (40-series) + positive flow
#maxsaldo = 3979     # value-driven negative flow (40 series)
#maxsaldo = 23444   # value-driven (42, big cities only)
#maxsaldo = 845      # value-driven 43

#maxsaldo = 3933     # dataset 24 inflow
maxsaldo = 4087     # dataset 24 outflow

# Получение данных
rawdata = pd.read_csv("superdataset/training ready/superdataset-24 outflow.csv")

rawdata = rawdata[rawdata.columns.drop('popsize')]
rawdata = rawdata[rawdata.columns.drop('beforeschool')]

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('outflow')])
datasetout = np.array(rawdata[['outflow']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.1, random_state=42)

# модель
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(trainin, trainout.ravel())

predtrain = model.predict(trainin)
errortrain = mean_absolute_error(trainout * maxsaldo, predtrain * maxsaldo)

predtest = model.predict(testin)
errortest = mean_absolute_error(testout * maxsaldo, predtest * maxsaldo)

a, b = MLS(testout, predtest)

migprop(model, testin, maxsaldo)

# ВЫВОД РЕЗУЛЬТАТОВ
# графики отклонения реального значения от прогнозируемого
scale = np.linspace(trainout.min() * maxsaldo, trainout.max() * maxsaldo, 100)
plt.scatter(testout * maxsaldo, predtest * maxsaldo, c='purple', alpha=.3, label='Testing set')
plt.plot(scale, scale, c='green', label='Ideal')
plt.plot(testout * maxsaldo, (testout * maxsaldo) * a + b, c='red', label='Bias of the model')
plt.axhline(0, c='k')
plt.axvline(0, c='k')
plt.xlabel('Actual values')
plt.ylabel('Predictied values')
plt.title("Net migration forecating")
plt.legend()
plt.show()

plt.plot(predtest[:100] * maxsaldo, label='Предсказание')
plt.plot(testout[:100] * maxsaldo, label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Миграционное сальдо")
plt.title("Прогноз на тестовой выборке")
plt.show()

#Корреляционная матрица Пирсона
cor = rawdata.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Значимость по критерию Джинни (сортировка, получение название признаков из датафрейма)
rawdata = rawdata[rawdata.columns.drop('outflow')]

important = model.feature_importances_

forplot = pd.DataFrame(data=important, index=rawdata.columns)
forplot = forplot.sort_values(by=[0])

plt.barh(forplot.index, forplot[0])
plt.show()

print("MAPE (train): ", errortrain)
print("MAPE (test): ", errortest)

