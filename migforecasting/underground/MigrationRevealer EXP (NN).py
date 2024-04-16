# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:50:13 2024

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
rawdata = pd.read_csv("citiesdataset-1.csv")
rawdata = np.array(rawdata)

resulttest = []
resulttrain = []

for k in range(20):
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
    model.fit(trainin, trainout, epochs=300, batch_size=5)
    
    predtrain = model.predict(trainin)
    errortrain = mean_absolute_percentage_error(trainout, predtrain)

    predtest = model.predict(testin)
    errortest = mean_absolute_percentage_error(testout, predtest)

    resulttrain.append(errortrain)
    resulttest.append(errortest)

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('testMAPE.xlsx')
resulttrain.to_excel('trainMAPE.xlsx')
    
    
    
    
    
    
    
    
    
    
    