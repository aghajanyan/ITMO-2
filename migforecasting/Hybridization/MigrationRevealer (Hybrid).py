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

maxsaldo = 26466

# Получение данных
rawdata = pd.read_csv("citiesdataset-NYDсor-4.csv")

rawdata = rawdata.sample(frac=1)  # перетасовка

# создание бинарного датасета для прогнозирования оттока/притока
rawdataclass = pd.DataFrame()
for i in range(rawdata.shape[0]):
    if rawdata.iloc[i, rawdata.shape[1] - 1] > 0:
        rawdataclass.append(rawdata[i])
        rawdataclass[i, len(rawdataclass) - 1] = 1
    else:
        rawdataclass.append(rawdata[i])
        rawdataclass[i, len(rawdataclass) - 1] = 0


# разбиение датасета на входные признаки и выходной результат (сальдо)
# 1 - для модели регресси, 2 - для модели классификатора
datasetin1 = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout1 = np.array(rawdata[['saldo']])

datasetin2 = np.array(rawdataclass[rawdataclass.columns.drop('saldo')])
datasetout2 = np.array(rawdataclass[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin1, testin1, trainout1, testout1 = train_test_split(datasetin1, datasetout1, test_size=0.2, random_state=42)
trainin2, testin2, trainout2, testout2 = train_test_split(datasetin2, datasetout2, test_size=0.2, random_state=42)

# модель 1
model1 = RandomForestRegressor(n_estimators=100, random_state=0)
model1.fit(trainin1, trainout1.ravel())

# модель 2
model2 = RandomForestClassifier(n_estimators=100, random_state=0)
model2.fit(trainin2, trainout2.ravel())

# объединение прогнозов и вычисление ошибки
predtrainreg = model1.predict(trainin1)
predtraindlass = model2.predict(trainin2)

#----заменить знаки в predtrainreg согласно predtraindlass

errortrain = mean_absolute_error(trainout * maxsaldo, predtrain * maxsaldo)

predtest = model.predict(testin)
errortest = mean_absolute_error(testout * maxsaldo, predtest * maxsaldo)
