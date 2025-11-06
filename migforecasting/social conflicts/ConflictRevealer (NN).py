import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import jax
os.environ["KERAS_BACKEND"] = "jax"
import keras
print(keras.__version__)

from keras.layers import Dense
from keras.models import Sequential


maxrisk = 3.873

# Получение данных
rawdata = pd.read_csv("datasets/agerow-superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21, top300, formodel-2).csv")

#rawdata = rawdata[rawdata.columns.drop('popsize')]
#rawdata = rawdata[rawdata.columns.drop('saldo')]

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (total score)
datasetin = np.array(rawdata[rawdata.columns.drop('risk')])
datasetout = np.array(rawdata[['risk']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

#модель
model = Sequential()
model.add(Dense(128, input_dim=28, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
history = model.fit(trainin, trainout, epochs=100, batch_size=10)

pred = model.predict(trainin)
pred1 = model.predict(testin)

scale = np.linspace(trainout.min() * maxrisk, trainout.max() * maxrisk, 100)
plt.scatter(testout * maxrisk, pred1 * maxrisk, c='black', alpha=.3, label='Testing set')
plt.plot([0, 4], [0, 4], ls='--', c='red', label='Ideal')
plt.xlabel('Actual values')
plt.ylabel('Predictied values')
plt.legend()
plt.show()

plt.plot(pred[:100], label='Предсказание')
plt.plot(testout[:100], label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Миграционное сальдо")
plt.title("Прогноз на тестовой выборке")
plt.show()

plt.plot(history.history['loss'])
plt.title("Learning process")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

trainloss = r2_score(trainout * maxrisk, pred * maxrisk)
testloss = r2_score(testout * maxrisk, pred1 * maxrisk)

print("Metrics on train: ", trainloss)
print("Metrics on test: ", testloss)

