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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout


# Получение данных
rawdata = pd.read_csv("datasets/citiesdataset-0.csv")
rawdata = np.array(rawdata)

np.random.shuffle(rawdata)

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

#модель
model = Sequential()
model.add(Dense(64, input_dim=18, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
history = model.fit(trainin, trainout, epochs=300, batch_size=5)

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

trainloss = mean_absolute_percentage_error(trainout, pred)
testloss = mean_absolute_percentage_error(testout, pred1)

print("MAPE on traning set: ", trainloss)
print("MAPE on testing set: ", testloss)

