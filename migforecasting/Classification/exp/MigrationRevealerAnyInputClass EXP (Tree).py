import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score


rawdata = pd.read_csv("citiesdataset-NYDCor-4 (CLASS).csv")

# Исключение из выборки отдельных признаков (отсутствуют у малых городов/райнов)
rawdata = rawdata.drop(['beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
                        'funds', 'companies', 'consnewapt', 'dollar'], axis=1)

#проверка точности предсказания на малых городах/территориях
village = pd.read_csv("input60NY (C).csv")

villagein = np.array(village[village.columns.drop('saldo')])
villageout = np.array(village[['saldo']])

resulttest = []
resulttrain = []
resultvillage = []
for k in range(50):
    rawdata = rawdata.sample(frac=1) # перетасовка

    # разбиение датасета на входные признаки и выходной результат (сальдо)
    datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
    datasetout = np.array(rawdata[['saldo']])

    # разбиение на обучающую и тестовую выборку
    trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

    # модель
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(trainin, trainout.ravel())

    predtrain = model.predict(trainin)
    errortrain = f1_score(trainout, predtrain)

    predtest = model.predict(testin)
    errortest = f1_score(testout, predtest)

    predvillage = model.predict(villagein)
    errorvillage = f1_score(villageout, predvillage)

    resulttrain.append(errortrain)
    resulttest.append(errortest)
    resultvillage.append(errorvillage)

    print('Итерация: ' + str(k))

resulttest = np.array(resulttest)
resulttrain = np.array(resulttrain)
resultvillage = np.array(resultvillage)

resulttest = pd.DataFrame(resulttest)
resulttrain = pd.DataFrame(resulttrain)
resultvillage = pd.DataFrame(resultvillage)

resulttest.to_excel('test.xlsx')
resulttrain.to_excel('train.xlsx')
resultvillage.to_excel('village.xlsx')