# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:32:23 2024

@author: Albert
"""

import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import jax
os.environ["KERAS_BACKEND"] = "jax"
import keras
print(keras.__version__)

from keras.layers import Dense
from keras.models import Sequential

#maxsaldo = 854     # dataset 24 (also balanced)
maxsaldo = 1009 

# Получение данных
rawdata = pd.read_csv("superdataset/training ready/superdataset-24 normbysoul.csv")

rawdata = rawdata[rawdata.columns.drop('popsize')]

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout = np.array(rawdata[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

#модель
model = Sequential()
model.add(Dense(64, input_dim=14, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss=keras.losses.MeanAbsoluteError())
history = model.fit(trainin, trainout, epochs=200, batch_size=8)

pred = model.predict(trainin)
pred1 = model.predict(testin)

plt.plot(pred[:100], label='Предсказание')
plt.plot(testout[:100], label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Миграционное сальдо")
plt.title("Прогноз на тестовой выборке")
plt.show()

plt.plot(history.history['loss'])
plt.title("Процесс обучения")
plt.xlabel("Номер эпохи")
plt.ylabel("Оценка отклонения")
plt.show()

trainloss = mean_absolute_error(trainout * maxsaldo, pred * maxsaldo)
testloss = mean_absolute_error(testout * maxsaldo, pred1 * maxsaldo)

print("Metrics on train: ", trainloss)
print("Metrics on test: ", testloss)

