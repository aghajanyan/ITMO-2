import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib


#Наименьшие квадраты для одной переменной
def MLS(x, y):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    n = len(x)
    sumx, sumy = sum(x), sum(y)
    sumx2 = sum([t * t for t in x])
    sumxy = sum([t * u for t, u in zip(x, y)])
    a = (n * sumxy - (sumx * sumy)) / (n * sumx2 - sumx * sumx)
    b = (sumy - a * sumx) / n
    return a, b

#maxrisk = 4.5
maxrisk = 3.873
# Получение данных
rawdata = pd.read_csv("datasets/avgage-superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21, top300, formodel-2).csv")

#rawdata = rawdata[rawdata.columns.drop('popsize')]
#rawdata = rawdata[rawdata.columns.drop('saldo')]

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (риск социального конфликта)
datasetin = np.array(rawdata[rawdata.columns.drop('risk')])
datasetout = np.array(rawdata[['risk']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

# модель
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(trainin, trainout.ravel())

predtrain = model.predict(trainin)
errortrain = r2_score(trainout * maxrisk, predtrain * maxrisk)

predtest = model.predict(testin)
errortest = r2_score(testout * maxrisk, predtest * maxrisk)

a, b = MLS(testout, predtest)


# ВЫВОД РЕЗУЛЬТАТОВ
# графики отклонения реального значения от прогнозируемого
scale = np.linspace(trainout.min() * maxrisk, trainout.max() * maxrisk, 100)
plt.scatter(testout * maxrisk, predtest * maxrisk, c='black', alpha=.3, label='Testing set')
plt.plot([-1, 4], [-1, 4], ls='--', c='red', label='Ideal')
#plt.plot(testout * maxrisk, (testout * maxrisk) * a + b, c='red', label='Bias of the model')
#plt.axhline(-1, c='k')
#plt.axvline(-1, c='k')
plt.xlabel('Actual values')
plt.ylabel('Predictied values')
plt.legend()
plt.show()

plt.plot(predtest[:100] * maxrisk, label='Предсказание')
plt.plot(testout[:100] * maxrisk, label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Вероятность социального конфликта")
plt.title("Прогноз на тестовой выборке")
plt.show()

"""
#Корреляционная матрица Пирсона
cor = rawdata.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
"""

# Значимость по критерию Джинни (сортировка, получение название признаков из датафрейма)
rawdata = rawdata[rawdata.columns.drop('risk')]

important = model.feature_importances_

forplot = pd.DataFrame(data=important, index=rawdata.columns)
forplot = forplot.sort_values(by=[0])

plt.barh(forplot.index, forplot[0])
plt.xlabel("Importance score")
plt.ylabel("Feature")
plt.show()

print("R2 (train): ", errortrain)
print("R2 (test): ", errortest)

# сохранение модели
#joblib.dump(model, ".joblib")