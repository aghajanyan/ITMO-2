import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


rawdata = pd.read_csv("dataset-0.csv")
rawdata = np.array(rawdata)

resulttest = []
resulttrain = []
for k in range(50):
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

    # модель
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(trainin, trainout)

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