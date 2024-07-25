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
    trainset = trainset[trainset.columns.drop('dollar')]
    return trainset


def normbyinf(trainset, rubfeatures):
    # умножить рублевые признаки на соответствующую долю инфляции
    inflation = pd.read_csv("inflation14.csv")
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


def normbyprod(trainset, rubfeatures):
    # разделить рублевые признаки на стоимость доллара
    prod = pd.read_csv("avgbeef.csv")
    trainset = trainset.merge(prod, on='year', how='left')
    d = trainset[['beefprice']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) / d.iloc[i, 0]
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    trainset = trainset[trainset.columns.drop('beefprice')]
    return trainset


# признаки для ценового нормирования
allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest',
                  'budincome', 'funds', 'naturesecure', 'factoriescap']

thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod', 'funds', 'factoriescap']

# получение и сортировка данных
rawdata = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/superdataset (full data).csv")
rawdata = rawdata.sort_values(by=['oktmo', 'year'])

dataset = []

rawdata = rawdata[rawdata.columns.drop('consnewapt')]
rawdata = rawdata[rawdata.columns.drop('foodservturnover')]
rawdata = rawdata[rawdata.columns.drop('invest')]
rawdata = rawdata[rawdata.columns.drop('budincome')]
rawdata = rawdata[rawdata.columns.drop('consnewareas')]
rawdata = rawdata[rawdata.columns.drop('cliniccap')]
#rawdata = rawdata[rawdata.columns.drop('popsize')]
rawdata = rawdata[rawdata.columns.drop('avgemployers')]
rawdata = rawdata[rawdata.columns.drop('avgsalary')]
rawdata = rawdata[rawdata.columns.drop('retailturnover')]
rawdata = rawdata[rawdata.columns.drop('livarea')]
rawdata = rawdata[rawdata.columns.drop('livestock')]
rawdata = rawdata[rawdata.columns.drop('harvest')]
rawdata = rawdata[rawdata.columns.drop('agrprod')]
rawdata = rawdata[rawdata.columns.drop('funds')]
rawdata = rawdata[rawdata.columns.drop('hospitals')]
rawdata = rawdata[rawdata.columns.drop('factoriescap')]
rawdata = rawdata[rawdata.columns.drop('naturesecure')]
rawdata = rawdata[rawdata.columns.drop('schoolnum')]
rawdata = rawdata[rawdata.columns.drop('beforeschool')]
rawdata = rawdata[rawdata.columns.drop('shoparea')]
rawdata = rawdata[rawdata.columns.drop('roadslen')]
rawdata = rawdata[rawdata.columns.drop('parks')]
rawdata = rawdata[rawdata.columns.drop('museums')]

"""
rawdata = rawdata[rawdata.columns.drop('consnewapt')]
rawdata = rawdata[rawdata.columns.drop('theatres')]
rawdata = rawdata[rawdata.columns.drop('museums')]
rawdata = rawdata[rawdata.columns.drop('parks')]
rawdata = rawdata[rawdata.columns.drop('cliniccap')]
rawdata = rawdata[rawdata.columns.drop('schoolnum')]
rawdata = rawdata[rawdata.columns.drop('naturesecure')]
rawdata = rawdata[rawdata.columns.drop('foodservturnover')]
rawdata = rawdata[rawdata.columns.drop('invest')]
rawdata = rawdata[rawdata.columns.drop('budincome')]
#rawdata = rawdata[rawdata.columns.drop('consnewareas')]
"""

# rawdata = rawdata.dropna(thresh=25)
rawdata = rawdata.dropna()

rawdata = rawdata.sort_values(by=['oktmo', 'year'])

#rawdata = normbyinf(rawdata, thisrubfeatures)

# удаление больших городов (население более 100 тысяч)
for index, row in rawdata.iterrows():
    if row['popsize'] > 100000:
        rawdata = rawdata.drop(index)

rawdata = rawdata[rawdata.columns.drop('popsize')]

"""C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately"""

library = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/library (allmun).csv")
library = library[library.columns.drop('name')]

rawdata = rawdata.merge(library, on=['oktmo', 'year'], how='left')

cultureorg = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/cultureorg (allmun).csv")
cultureorg = cultureorg[cultureorg.columns.drop('name')]

rawdata = rawdata.merge(cultureorg, on=['oktmo', 'year'], how='left')

musartschool = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/musartschool (allmun).csv")
musartschool = musartschool[musartschool.columns.drop('name')]

rawdata = rawdata.merge(musartschool, on=['oktmo', 'year'], how='left')


rawdata = rawdata.dropna()

"""
# удаление полностью нулевых примеров
for index, row in rawdata.iterrows():
    if (row['foodseats'] == 0 and row['sportsvenue'] == 0 and row['servicesnum'] == 0 and row['museums'] == 0 and
            row['parks'] == 0 and row['theatres'] == 0):
        rawdata = rawdata.drop(index)

"""

"""
# удаление муниципальных районов (только города)
for index, row in rawdata.iterrows():
    tmp = row['name'].split()
    if len(tmp) > 1:
        if tmp[0] != 'город' and tmp[0] != 'город-курорт' and tmp[0] != 'город-герой':
            rawdata = rawdata.drop(index)
"""
# удалить из датасета отрицательное/положительное сальдо
count = 0
for index, row in rawdata.iterrows():
    if row['saldo'] < 0:
        rawdata = rawdata.drop(index)

# убрать отрицательный знак
#rawdata['saldo'] = rawdata['saldo'].abs()

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
# анализ признаков датасета
features = ['saldo', 'foodseats', 'sportsvenue', 'servicesnum', 'museums', 'parks', 'theatres',
            'library', 'cultureorg', 'musartschool']

examples = pd.DataFrame(examples, columns=features)

avg = examples.mean()
maxmax = examples.max()
minmin = examples.min()

x = 0
count = []
for k in range(0, examples.shape[1]):
    for i in range(examples.shape[0]):
        if examples.iloc[i, k] == 0:
            x+=1
    count.append(examples.columns[k])
    count.append(x)
    x = 0
"""

# нормализация от 0 до 1
examples = normbymax(examples)

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
"""
features = ['saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'consnewareas', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'funds', 'hospitals', 'beforeschool', 'factoriescap']
"""
features = ['saldo', 'foodseats', 'sportsvenue', 'servicesnum', 'museums', 'parks', 'theatres',
            'library', 'cultureorg', 'musartschool']

examples = pd.DataFrame(examples, columns=features)

examples.to_csv("superdataset-41 (positive flow).csv", index=False)

print('Done')
