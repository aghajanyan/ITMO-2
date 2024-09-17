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

model1error = mean_absolute_error(trainout1, predinflow) * maxsaldoIn
print('Обуч. приток: ', model1error)

# модель 2
model2 = RandomForestRegressor(n_estimators=100, random_state=0)
model2.fit(trainin, trainout2.ravel())
predoutflow = model2.predict(trainin)

model2error = mean_absolute_error(trainout2, predoutflow) * maxsaldoOut
print('Обуч. отток: ', model2error)

errortrain = mean_absolute_error(((trainout1 * maxsaldoIn) - (trainout2 * maxsaldoOut)),
                                 ((predinflow * maxsaldoIn) - (predoutflow * maxsaldoOut)))

print('Обучение сальдо: ', errortrain)

predinflow = model1.predict(testin)
predoutflow = model2.predict(testin)

errortest = mean_absolute_error(((testout1 * maxsaldoIn) - (testout2 * maxsaldoOut)),
                                ((predinflow * maxsaldoIn) - (predoutflow * maxsaldoOut)))
print('Тест сальдо: ', errortest)

