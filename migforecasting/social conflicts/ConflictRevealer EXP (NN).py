import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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
rawdata = pd.read_csv("datasets/superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21, top300, formodel-2).csv")

rawdata = rawdata[rawdata.columns.drop('popsize')]
rawdata = rawdata[rawdata.columns.drop('saldo')]

resulttest = []
resulttrain = []

for k in range(20):
    rawdata = rawdata.sample(frac=1)  # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop('risk')])
    datasetout = np.array(rawdata[['risk']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)
    
    #модель
    model = Sequential()
    model.add(Dense(128, input_dim=15, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    history = model.fit(trainin, trainout, epochs=100, batch_size=10)
    
    predtrain = model.predict(trainin)
    errortrain = mean_squared_error(trainout * maxrisk, predtrain * maxrisk)

    predtest = model.predict(testin)
    errortest = mean_squared_error(testout * maxrisk, predtest * maxrisk)

    resulttrain.append(errortrain)
    resulttest.append(errortest)

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test.xlsx')
resulttrain.to_excel('train.xlsx')
    
    
    
    
    
    
    
    
    
    
    