import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def normbymax(trainset):
    tmpp = []
    for k in range(len(trainset[0])):
        maxi = trainset[0][k]
        for i in range(len(trainset)):
            if (maxi < trainset[i][k]):
                maxi = trainset[i][k]

        tmpp.append(maxi)

        for j in range(len(trainset)):
            trainset[j][k] = trainset[j][k] / maxi
    return trainset


def normbydollar(trainset, rubfeatures):
    # разделить рублевые признаки на стоимость доллара
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


def normbyinf(trainset, rubfeatures):
    # умножить рублевые признаки на соответствующую долю инфляции
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


def normbyoil(trainset, rubfeatures):
    # умножить рублевые признаки на цену за нефть как долю процента
    oil2 = pd.read_csv("oilpricesavg.csv")
    trainset = trainset.merge(oil2, on='year', how='left')
    o = trainset[['oil']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                oilnorm = o.iloc[i, 0] / 100
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) * oilnorm
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    trainset = trainset[trainset.columns.drop('oil')]
    return trainset


# признаки для ценового нормирования
allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest',
                  'budincome', 'funds', 'naturesecure', 'factoriescap']

thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod', 'invest', 'budincome', 'funds', 'factoriescap']

# получение и сортировка данных
rawdata = pd.read_csv(
    "C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/datasets/superdataset/superdataset (full data).csv")
rawdata = rawdata.sort_values(by=['oktmo', 'year'])

dataset = []
# формирование полного датасета, но только с примерами, где все признаки не NaN

rawdata = rawdata[rawdata.columns.drop('consnewapt')]
rawdata = rawdata[rawdata.columns.drop('theatres')]
rawdata = rawdata[rawdata.columns.drop('museums')]
rawdata = rawdata[rawdata.columns.drop('parks')]
rawdata = rawdata[rawdata.columns.drop('cliniccap')]
rawdata = rawdata[rawdata.columns.drop('schoolnum')]
rawdata = rawdata[rawdata.columns.drop('naturesecure')]
rawdata = rawdata[rawdata.columns.drop('foodservturnover')]

# rawdata = rawdata.dropna(thresh=25)
rawdata = rawdata.dropna()

rawdata = rawdata.sort_values(by=['oktmo', 'year'])

#rawdata = normbydollar(rawdata, thisrubfeatures)

examples = []
# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata) - 1):
    if rawdata.iloc[i, 0] == rawdata.iloc[i + 1, 0]:
        rawdata.iloc[i, 3] = rawdata.iloc[i + 1, 3]
        examples.append(rawdata.iloc[i])

examples = np.array(examples)

examples = np.delete(examples, 2, 1)  # удаляем год
examples = np.delete(examples, 1, 1)  # удаляем название мун. образования
examples = np.delete(examples, 0, 1)  # удаляем октмо

"""
# подсчет количества NaNов у признака
x = 0
count = []
for k in range(5, rawdata.shape[1]):
    for i in range(rawdata.shape[0]):
        if rawdata.iloc[i, k] != rawdata.iloc[i, k]:
            x+=1
    count.append(rawdata.columns[k])
    count.append(x)
    x = 0
"""

features = ['saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'consnewareas', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'invest', 'budincome', 'funds',
            'hospitals', 'beforeschool', 'factoriescap']

examples = pd.DataFrame(examples, columns=features)

examples.to_csv("superdataset-00.csv", index=False)


print('Done')
