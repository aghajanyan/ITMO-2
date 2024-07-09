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

maxsaldoP = 10001    # dataset 20 (also positive flow)
maxsaldoN = 2854     # negative flow (dataset 20)

positive = pd.read_csv("superdataset-21 (positive flow).csv")
negative = pd.read_csv("superdataset-21 (negative flow).csv")
dataclass = []
alldata = []

negative = negative[negative.columns.drop('consnewareas')]
positive = positive[positive.columns.drop('consnewareas')]

negative = negative.sample(frac=1)  # перетасовка
positive = positive.sample(frac=1)

# разбиение датасета на входные признаки и выходной результат (сальдо)
# 1 - для оттока (негатив), 2 - для притока (позитив)
datasetin1 = np.array(negative[negative.columns.drop('saldo')])
datasetin2 = np.array(positive[positive.columns.drop('saldo')])
datasetout1 = np.array(negative[['saldo']])
datasetout2 = np.array(positive[['saldo']])


#объединение датасетов с возвращением знака
