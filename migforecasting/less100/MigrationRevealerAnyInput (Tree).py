import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''Наименьшие квадраты для одной переменной'''
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

maxsaldo = 26466

# Получение данных
rawdata = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/datasets/citiesdataset-NYDCor-4.csv")

# Исключение из выборки отдельных признаков (отсутствуют у малых городов/райнов)
rawdata = rawdata.drop(['beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
                        'funds', 'companies', 'consnewapt', 'conscap', 'livarea', 'consnewareas', 'dollar'], axis=1)

rawdata = rawdata.sample(frac=1)  # перетасовка

# разбиение датасета на входные признаки и выходной результат (сальдо)
datasetin = np.array(rawdata[rawdata.columns.drop('saldo')])
datasetout = np.array(rawdata[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=0.2, random_state=42)

# модель
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(trainin, trainout.ravel())

predtrain = model.predict(trainin)
errortrain = mean_absolute_percentage_error(trainout * maxsaldo, predtrain * maxsaldo)

predtest = model.predict(testin)
errortest = mean_absolute_percentage_error(testout * maxsaldo, predtest * maxsaldo)

a, b = MLS(testout, predtest)

# вывод результатов
scale = np.linspace(trainout.min() * maxsaldo, trainout.max() * maxsaldo, 100)
plt.scatter(testout * maxsaldo, predtest * maxsaldo, c='purple', alpha=.3, label='Тестовая выборка')
plt.plot(scale, scale, c='green', label='Идеал')
plt.plot(testout * maxsaldo, (testout * maxsaldo) * a + b, c='red', label='Смещение модели')
plt.axhline(0, c='k')
plt.axvline(0, c='k')
plt.xlabel('Реальное значение')
plt.ylabel('Предсказание')
plt.legend()
plt.show()

plt.plot(predtest[:100] * maxsaldo, label='Предсказание')
plt.plot(testout[:100] * maxsaldo, label='Реальное значение')
plt.legend(loc='upper left')
plt.xlabel("Номер теста")
plt.ylabel("Миграционное сальдо")
plt.title("Прогноз на тестовой выборке")
plt.show()

"""
features = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
            'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
            'invests', 'funds', 'companies', 'factoriescap',
            'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
            'foodservturnover']
            
            'Широта', 'Долгота', 'Доллар'

features = ['Числ. насл.', 'Ср. кол-во. раб.', 'Безраб.', 'Ср. з/п', 'Площ. на чел.',
            'Дошкол.', 'Врачей на чел.', 'Коек на чел.', 'Мощ. клиник',
            'Инвест.', 'Фонды', 'Предприятия', 'Мощ. промыш.',
            'Объемы строит.', 'Постр. жил. площ.', 'Постр. кварт.', 'Оборот розницы',
            'Оборот общепит.', 'Широта', 'Долгота', 'dollar', 'Oil']
"""
features = ['Числ. насл.', 'Ср. кол-во. раб.', 'Безраб.', 'Ср. з/п',
            'Инвест.', 'Мощ. промыш.', 'Оборот розницы',
            'Оборот общепит.', 'Широта', 'Долгота']

important = model.feature_importances_

plt.barh(features, important)
plt.show()

print("MAPE (train): ", errortrain)
print("MAPE (test): ", errortest)

#проверка точности предсказания на малых городах/территориях
village = pd.read_csv("moreinputNY.csv")

villagein = np.array(village[village.columns.drop('saldo')])
villageout = np.array(village[['saldo']])

predvillage = model.predict(villagein)
errorvillage = mean_absolute_percentage_error(villageout * maxsaldo, predvillage * maxsaldo)

print("MAPE for input cities: ", errorvillage)

predvillage = np.array(predvillage * maxsaldo)
predvillage = pd.DataFrame(predvillage)

predvillage.to_excel("output.xlsx")


