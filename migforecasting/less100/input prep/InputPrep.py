import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

allmax = {
        'popsize': 1625.6,
        'avgemployers': 475.8,
        'unemployed': 44134.0,
        'avgsalary': 65935.7371,
        'livarea': 69.6,
        # 'beforeschool': 50640.0,
        # 'docsperpop': 146.1,
        # 'bedsperpop': 252.0,
        # 'cliniccap': 1156.4,
        'invests': 1646.8895179039303,
        # 'funds': 89.9,
        # 'companies': 209412.0,
        'factoriescap': 567115.8942,
        'conscap': 106568.55884,
        'consnewareas': 7563.0,
        # 'consnewapt': 42801.0,
        'retailturnover': 154512.08200999998,
        'foodservturnover': 16055.4,
        'lat': 69.38294581595048,
        'lon': 177.49228362395198,
        # 'oil': 97.98,
        'saldo': 26466.0
    }

def normbymax(trainset):    # нормализовать входные признаки по максимальным значениям датасета
    for k in allmax:
        tmp = trainset[[k]]
        for i in range(len(tmp)):
            tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) / float(allmax[k])
        trainset[k] = tmp
        tmp = pd.DataFrame(None)
    return trainset


def normbyinf(trainset):  # умножить рублевые признаки на соответствующую долю инфляции
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
rawdata = pd.read_excel("inputNY.xlsx")
rawdata = rawdata.sort_values(by=['name', 'year'])

rawdata = normbyinf(rawdata)
rawdata = normbymax(rawdata)

examples = []

# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata)):
    #if rawdata.iloc[i, 0] == rawdata.iloc[i + 1, 0]:
        #rawdata.iloc[i, rawdata.shape[1] - 1] = rawdata.iloc[i + 1, rawdata.shape[1] - 1]
    examples.append(rawdata.iloc[i])

examples = np.delete(examples, 1, 1)  # удаляем год
examples = np.delete(examples, 0, 1)  # удаляем название городов

# запись в csv
titles = allmax.keys()

examples = pd.DataFrame(examples, columns=titles)

examples.to_csv("inputNY.csv", index=False)

print('Done')