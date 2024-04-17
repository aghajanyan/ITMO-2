import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class Normalization:
    def normbymax(trainset):
        for k in range(len(trainset[0])):
            maxi = trainset[0][k]
            for i in range(len(trainset)):
                if (maxi < trainset[i][k]):
                    maxi = trainset[i][k]

            for j in range(len(trainset)):
                trainset[j][k] = trainset[j][k] / maxi

#получение и сортировка данных
rawdata = pd.read_csv("citiesdataset 10-21 (+y).csv")
rawdata = rawdata.sort_values(by=['name', 'year'])

examples = []

# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata) - 1):
    if rawdata.iloc[i, 0] == rawdata.iloc[i + 1, 0]:
        rawdata.iloc[i, 20] = rawdata.iloc[i + 1, 20]
        examples.append(rawdata.iloc[i])

examples = np.array(examples)

#удаляем из датасета Москву и Питер
i = 0
while i < len(examples):
    if examples[i, 0] == 'Москва' or examples[i, 0] == 'Санкт-Петербург':
        examples = np.delete(examples, i, 0)
        i-=1
    else:
        i+=1

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
                tmpavg+= float(examples[i, k])
                count+=1
            except ValueError:
                tmpavg+=0
        else:
            tmpavg+=0
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

Normalization.normbymax(examples)

# запись в csv
titles = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
          'invests', 'funds', 'companies', 'factoriescap',
          'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
          'foodservturnover', 'saldo']

examples = pd.DataFrame(examples, columns=titles)

examples.to_csv("citiesdataset-NY-2.csv", index=False)

print('Done')