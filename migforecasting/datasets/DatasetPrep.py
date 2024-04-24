import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def normbymax(trainset):
    for k in range(len(trainset[0])):
        maxi = trainset[0][k]
        for i in range(len(trainset)):
            if (maxi < trainset[i][k]):
                maxi = trainset[i][k]

        for j in range(len(trainset)):
            trainset[j][k] = trainset[j][k] / maxi
    return trainset


def normbydollar(trainset):
    # разделить рублевые признаки на стоимость доллара
    rubfeatures = ['avgsalary', 'invests', 'factoriescap', 'conscap', 'retailturnover', 'foodservturnover']
    dollar = pd.read_csv("dollaravg.csv")
    trainset = trainset.merge(dollar, on='year', how='left')
    d = trainset[['dollar']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) / d.iloc[i, 0]
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    return trainset


def normbyinf(trainset):
    # умножить рублевые признаки на соответствующую долю инфляции
    rubfeatures = ['avgsalary', 'invests', 'factoriescap', 'conscap', 'retailturnover', 'foodservturnover']
    inflation = pd.read_csv("inflation1.csv")
    trainset = trainset.merge(inflation, on='year', how='left')
    inf = trainset[['inf']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                infnorm = 1 - (inf.iloc[i, 0] / 100)
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) * infnorm
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    trainset = trainset[trainset.columns.drop('inf')]
    return trainset



# получение и сортировка данных
rawdata = pd.read_csv("citiesdataset 10-21.csv")
rawdata = rawdata.sort_values(by=['name', 'year'])

# добавление координат
coordinates = pd.read_csv("coordinates.csv")
rawdata = rawdata.merge(coordinates, on='name', how='left')

dollar = pd.read_csv("dollaravg.csv")
rawdata = rawdata.merge(dollar, on='year', how='left')

# сальдо в конец таблицы
saldo = rawdata[['saldo']]
rawdata = rawdata[rawdata.columns.drop('saldo')]
rawdata = pd.concat([rawdata, saldo], axis=1)

# rawdata = normbydollar(rawdata)
rawdata = normbyinf(rawdata)
examples = []

# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata) - 1):
    #if rawdata.iloc[i, 0] == rawdata.iloc[i + 1, 0]:
        #rawdata.iloc[i, 20] = rawdata.iloc[i + 1, 20]
        examples.append(rawdata.iloc[i])

examples = np.array(examples)

# удаляем из датасета Москву и Питер
i = 0
while i < len(examples):
    if examples[i, 0] == 'Москва' or examples[i, 0] == 'Санкт-Петербург':
        examples = np.delete(examples, i, 0)
        i -= 1
    else:
        i += 1

examples = np.delete(examples, 1, 1)  # удаляем год
examples = np.delete(examples, 0, 1)  # удаляем название городов

# вычисление среднего для каждого признака
avg = []
tmpavg = 0
count = 0
for k in range(len(examples[1])):
    for i in range(len(examples)):
        if examples[i, k] == examples[i, k]:  # проверка NaN
            try:
                tmpavg += float(examples[i, k])
                count += 1
            except ValueError:
                tmpavg += 0
        else:
            tmpavg += 0
    avg.append(tmpavg / count)
    tmpavg = 0
    count = 0

# перевод из текста в число (заменить средним при невозможности конвертации)
i = 0
while i < len(examples):
    for j in range(len(examples[1])):
        if examples[i, j] == examples[i, j]:  # проверка NaN
            try:
                examples[i, j] = float(examples[i, j])
            except ValueError:
                examples[i, j] = avg[j]
                i -= 1
                break
        else:
            examples[i, j] = avg[j]
            i -= 1
            break
    i += 1

examples = normbymax(examples)

# запись в csv
titles = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
          'invests', 'funds', 'companies', 'factoriescap',
          'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
          'foodservturnover', 'lon', 'lat', 'dollar', 'saldo']

examples = pd.DataFrame(examples, columns=titles)

examples.to_csv("citiesdataset-DCor-4.csv", index=False)

print('Done')
