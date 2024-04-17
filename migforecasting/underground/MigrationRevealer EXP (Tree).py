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


rawdata = pd.read_csv("datasets/citiesdataset-NY-1.csv")

resulttest = []
resulttrain = []
for k in range(50):
    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
    datasetout = np.array(rawdata[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.1, random_state=42)

    # модель
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(trainin, trainout.ravel())

    predtrain = model.predict(trainin)
    errortrain = mean_absolute_percentage_error(trainout, predtrain)

    predtest = model.predict(testin)
    errortest = mean_absolute_percentage_error(testout, predtest)

    resulttrain.append(errortrain)
    resulttest.append(errortest)

    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('testMAPE.xlsx')
resulttrain.to_excel('trainMAPE.xlsx')