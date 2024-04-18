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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout


# Получение данных
rawdata = pd.read_csv("datasets/citiesdataset-NY-2.csv")

resulttest = []
resulttrain = []
for k in range(20):
    rawdata = rawdata.sample(frac=1)  # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
    datasetout = np.array(rawdata[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)
    
    #модель
    model = Sequential()
    model.add(Dense(64, input_dim=18, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    model.fit(trainin, trainout, epochs=300, batch_size=5)
    
    predtrain = model.predict(trainin)
    errortrain = mean_squared_error(trainout, predtrain)

    predtest = model.predict(testin)
    errortest = mean_squared_error(testout, predtest)

    resulttrain.append(errortrain)
    resulttest.append(errortest)

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('testMAPE.xlsx')
resulttrain.to_excel('trainMAPE.xlsx')
    
    
    
    
    
    
    
    
    
    
    