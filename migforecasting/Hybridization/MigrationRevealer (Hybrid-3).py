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

#перевод из нормализованного диапазона в реальное
maxsaldoP = 10001    # dataset 20 (also positive flow)
maxsaldoN = 2854     # negative flow (dataset 20)

positive = pd.read_csv("superdataset-21 (positive flow).csv")
negative = pd.read_csv("superdataset-21 (negative flow).csv")

negative = negative[negative.columns.drop('consnewareas')]
positive = positive[positive.columns.drop('consnewareas')]

# перетасовка
negative = negative.sample(frac=1)
positive = positive.sample(frac=1)

#одинаковая нормализация сальдо
#for i in range(len(negative)):
#    negative.iloc[i, 0] = negative.iloc[i, 0] * maxsaldoN
#    negative.iloc[i, 0] = negative.iloc[i, 0] / maxsaldoP

# разбиение датасета на входные признаки и выходной результат (сальдо)
# 1 - для оттока (негатив), 2 - для притока (позитив), 3 - для классификатора
datasetin1 = np.array(negative[negative.columns.drop('saldo')])
datasetin2 = np.array(positive[positive.columns.drop('saldo')])
datasetout1 = np.array(negative[['saldo']])
datasetout2 = np.array(positive[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin1, testin1, trainout1, testout1 = train_test_split(datasetin1, datasetout1, test_size=0.2, random_state=146)
trainin2, testin2, trainout2, testout2 = train_test_split(datasetin2, datasetout2, test_size=0.2, random_state=146)

#подготовка входных и выходных данных для обучения классификатора
#объединить входы негатива и позитива
classdata, trainout3 = [], []
for i in range(trainin2.shape[0]):
    classdata.append(np.append(trainin1[i], 0))  # 0 - для отрицательного сальдо
    classdata.append(np.append(trainin2[i], 1))  # 1 - для подожительного сальдо

classdata = pd.DataFrame(classdata)
classdata = classdata.sample(frac=1)    # перетасовка

# разделение входных и выходных данных
trainout3 = classdata[[classdata.shape[1] - 1]]
classdata = classdata[classdata.columns.drop(classdata.shape[1] - 1)]
trainin3 = classdata

trainin3 = np.array(trainin3)
trainout3 = np.array(trainout3)

#для тестовой выборки тасовка не требуется
testin3, testout3 = [], []
for i in range(testin2.shape[0]):
    testin3.append(testin1[i])
    testout3.append(0)
    testin3.append(testin2[i])
    testout3.append(1)

testin3 = np.array(testin3)
testout3 = np.array(testout3)

