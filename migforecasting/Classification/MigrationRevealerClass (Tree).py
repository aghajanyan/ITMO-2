
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score



# Получение данных
rawdata = pd.read_csv("citiesdataset-NYDcor-4 (CLASS).csv")

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout = np.array(rawdata[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

# модель
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(trainin, trainout.ravel())

predtest = model.predict(testin)

fone = f1_score(testout, predtest)
print(fone)

fpr, tpr, thresholds = roc_curve(testout, predtest)
aucscore = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % aucscore)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('Прогноз оттока или притока миграции (тестовый сет)')
plt.legend(loc="lower right")
plt.show()