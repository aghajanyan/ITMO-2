import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


rawdata = pd.read_csv("datasets/superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21, top300, formodel-2).csv")

rawdata = rawdata[rawdata.columns.drop('popsize')]
rawdata = rawdata[rawdata.columns.drop('saldo')]

resulttest = []
resulttrain = []

#maxrisk = 4.5
maxrisk = 3.873

signif = []
n = 50
for k in range(n):
    rawdata = rawdata.sample(frac=1) # перетасовка

    # split the dataset into input and output
    datasetin = np.array(rawdata[rawdata.columns.drop('risk')])
    datasetout = np.array(rawdata[['risk']])

    # split the learning set on train-set and test-set
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

    # the model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(trainin, trainout.ravel())

    # error score (deviation between predicted and real values)
    predtrain = model.predict(trainin)
    errortrain = mean_absolute_error(trainout * maxrisk, predtrain * maxrisk)

    predtest = model.predict(testin)
    errortest = mean_absolute_error(testout * maxrisk, predtest * maxrisk)

    # save the error score
    resulttrain.append(errortrain)
    resulttest.append(errortest)

    print('Iteration: ' + str(k))

    # cumulative feature importances
    important = model.feature_importances_
    for i, v in enumerate(important):
        if k == 0:
            signif.append(v)
        else:
            signif[i]+= v

# average feature importances
for i in range(len(signif)):
    signif[i] = signif[i] / n

signif = np.array(signif)
signif = pd.DataFrame(signif)
signif.to_excel('feature significance.xlsx')

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)

resulttest.to_excel('test-data.xlsx')
resulttrain.to_excel('train-data.xlsx')