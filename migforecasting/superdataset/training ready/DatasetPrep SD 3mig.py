import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import copy


# нормализация знечений признаков от 0 до 1 (с сохранением файла с нормализаторами (макс.))
def normbymax(trainset):
    tmpp = []
    for k in range(0, len(trainset[0])):
        maxi = trainset[0][k]
        for i in range(len(trainset)):
            if (maxi < trainset[i][k]):
                maxi = trainset[i][k]

        tmpp.append(maxi)

        for j in range(len(trainset)):
            trainset[j][k] = trainset[j][k] / maxi

    features = ['reg', 'interreg', 'internat', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
            'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest',
            'agrprod', 'hospitals', 'beforeschool']

    tmpp = np.array(tmpp)
    tmpp = pd.DataFrame([tmpp], columns=features)
    tmpp.to_csv("fornorm 24 3mig 2Y.csv", index=False)

    return trainset


# разделить рублевые признаки на стоимость доллара
def normbydollar(trainset, rubfeatures):
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


# умножить рублевые признаки на соответствующую долю инфляции
def normbyinf(trainset, rubfeatures):
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


# умножить рублевые признаки на цену за нефть как долю процента
def normbyoil(trainset, rubfeatures):
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


# разделить рублевые признаки на стоимость продукта
def normbyprod(trainset, rubfeatures):
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


# IQR от Кирилла
def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    data = copy.deepcopy(data)

    for col in data.columns:
        if col != 'popsize':
            data = data[(data[col] < np.quantile(data[col], 0.75) + 4 * sts.iqr(data[col])) &
                        (data[col] > np.quantile(data[col], 0.25) - 4 * sts.iqr(data[col]))]

    return data


# анализ признаков датасета
def featuresanalysis(examples):
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
                x += 1
        count.append(examples.columns[k])
        count.append(x)
        x = 0


# удаление примеров с нулевыми значениями признаков
def delifzero(data):
    for index, row in data.iterrows():
        if (row['foodseats'] == 0 and row['sportsvenue'] == 0 and row['servicesnum'] == 0 and row['museums'] == 0 and
                row['parks'] == 0 and row['theatres'] == 0):
            data = data.drop(index)


# удаление из датасета определенных мун. образований
def onlycertainmun(data):
    mun = False
    for index, row in data.iterrows():
        tmp = row['name'].split()
        for i in range(len(tmp)):
            if tmp[i] == 'муниципальный' or tmp[i] == 'Муниципальный':
                mun = True
        if mun:
            data = data.drop(index)
            mun = False
        else:
            mun = False

    return data


# удалить из датасета отрицательное/положительное сальдо
def delnegorpos(data, sign):
    for index, row in data.iterrows():
        if sign == 0:
            if row['saldo'] < 0:
                data = data.drop(index)
        else:
            if row['saldo'] > 0:
                data = data.drop(index)

    # убрать отрицательный знак
    #data['saldo'] = data['saldo'].abs()

    return data


# подсчет количества NaNов у признака
def nannumber(data):
    x = 0
    count = []
    for k in range(5, data.shape[1]):
        for i in range(data.shape[0]):
            if data.iloc[i, k] != data.iloc[i, k]:
                x += 1
        count.append(data.columns[k])
        count.append(x)
        x = 0


# нормирование факторов на душу населения для всего датасета
def normpersoulalldata(data):
    # факторы для нормирования
    features = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool', 'factoriescap']

    for a in features:
        data[a] = data[a] / data['popsize']


# признаки для ценового нормирования
allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest',
                  'budincome', 'funds', 'naturesecure', 'factoriescap']

thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']

# получение и сортировка данных
rawdata = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/superdataset (full data).csv")
rawdata = rawdata.sort_values(by=['oktmo', 'year'])

rawdata = rawdata[rawdata.columns.drop('saldo')]

#migtype = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/saldo reg (allmun).csv")
#migtype = migtype[migtype.columns.drop('name')]

#rawdata = rawdata.merge(migtype, on=['oktmo', 'year'], how='left')

reg = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/saldo reg (allmun).csv")
reg = reg[reg.columns.drop('name')]
reg.columns = ['oktmo', 'year', 'reg']

interreg = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/saldo interreg (allmun).csv")
interreg = interreg[interreg.columns.drop('name')]
interreg.columns = ['oktmo', 'year', 'interreg']

internat = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/saldo internat (allmun).csv")
internat = internat[internat.columns.drop('name')]
internat.columns = ['oktmo', 'year', 'internat']

rawdata = rawdata.merge(reg, on=['oktmo', 'year'], how='left')
rawdata = rawdata.merge(interreg, on=['oktmo', 'year'], how='left')
rawdata = rawdata.merge(internat, on=['oktmo', 'year'], how='left')

#outflow = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/outflow (allmun).csv")
#inflow = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/inflow (allmun).csv")

#outflow = outflow[outflow.columns.drop('name')]
#inflow = inflow[inflow.columns.drop('name')]

#rawdata = rawdata.merge(outflow, on=['oktmo', 'year'], how='left')
#rawdata = rawdata.merge(inflow, on=['oktmo', 'year'], how='left')
#rawdata = rawdata[rawdata.columns.drop('saldo')]

dataset = []
"""
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
rawdata = rawdata[rawdata.columns.drop('theatres')]

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
rawdata = rawdata[rawdata.columns.drop('consnewareas')]
#rawdata = rawdata[rawdata.columns.drop('shoparea')]
#rawdata = rawdata[rawdata.columns.drop('servicesnum')]
rawdata = rawdata[rawdata.columns.drop('funds')]
rawdata = rawdata[rawdata.columns.drop('factoriescap')]


# rawdata = rawdata.dropna(thresh=25)
rawdata = rawdata.dropna()

rawdata = rawdata.sort_values(by=['oktmo', 'year'])

cols = ['oktmo', 'name', 'year', 'reg', 'interreg', 'internat', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
        'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
        'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']

rawdata = rawdata[cols]


#rawdata = normbyinf(rawdata, thisrubfeatures)

# визуальный анализ распределения конкретного признака
#x = [1] * len(rawdata['popsize'])
#y = list(rawdata['popsize'])

#plt.plot(x, y, 'o', mfc='none', color='black')
#plt.show()

#rawdata = onlycertainmun(rawdata)

"""
# удаление больших городов (население более 100 тысяч)
for index, row in rawdata.iterrows():
    if row['popsize'] > 100000:
        rawdata = rawdata.drop(index)
"""
"""
# выборка только за определенный год
for index, row in rawdata.iterrows():
    if row['year'] != 2022:
        rawdata = rawdata.drop(index)
"""

#rawdata = delnegorpos(rawdata)

"""
rawdata = rawdata[rawdata.columns.drop('popsize')]

#C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately

library = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/library (allmun).csv")
library = library[library.columns.drop('name')]

rawdata = rawdata.merge(library, on=['oktmo', 'year'], how='left')

cultureorg = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/cultureorg (allmun).csv")
cultureorg = cultureorg[cultureorg.columns.drop('name')]

rawdata = rawdata.merge(cultureorg, on=['oktmo', 'year'], how='left')

#musartschool = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/musartschool (allmun).csv")
#musartschool = musartschool[musartschool.columns.drop('name')]

#rawdata = rawdata.merge(musartschool, on=['oktmo', 'year'], how='left')

rawdata = rawdata.dropna()


goodcompincome = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/features separately/goodcompincome (allmun).csv")
goodcompincome = goodcompincome[goodcompincome.columns.drop('name')]

rawdata = rawdata.merge(goodcompincome, on=['oktmo', 'year'], how='left')
rawdata = rawdata.dropna()
"""

rawdata = normbyinf(rawdata, thisrubfeatures)

examples = []
ny = 2 # на сколько лет вперед сальдо в качестве выходного результата
# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata) - ny):
    if rawdata.iloc[i, 0] == rawdata.iloc[i + ny, 0]:
        if rawdata.iloc[i, 2] + ny == rawdata.iloc[i + ny, 2]:  # прогноз только на год вперед
            rawdata.iloc[i, 3] = rawdata.iloc[i + ny, 3]     # сдвигаем inflow / saldo / reg
            rawdata.iloc[i, 4] = rawdata.iloc[i + ny, 4]     # сдвигаем outflow / interreg
            rawdata.iloc[i, 5] = rawdata.iloc[i + ny, 5]     # сдвигаем internat
            examples.append(rawdata.iloc[i])

examples = np.array(examples)

examples = np.delete(examples, 2, 1)  # удаляем год
examples = np.delete(examples, 1, 1)  # удаляем название мун. образования
examples = np.delete(examples, 0, 1)  # удаляем октмо

features = ['reg', 'interreg', 'internat', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
            'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest',
            'agrprod', 'hospitals', 'beforeschool']

examples = pd.DataFrame(examples, columns=features)

# нормирование на душу населения
#normpersoulalldata(examples)

examples = remove_outliers(examples)

"""
# анализ признаков
avg = examples.mean()
minmin = examples.min()
maxmax = examples.max()

avg = pd.DataFrame(avg, columns=['avg'])
minmin = pd.DataFrame(minmin, columns=['min'])
maxmax = pd.DataFrame(maxmax, columns=['max'])
avg = avg.join(minmin)
avg = avg.join(maxmax)
avg.to_excel('FeatureAalysis.xlsx')
"""
"""
# создание сбалансированной выборки (одинаковое количество положительные и отрицательных примеров)
examples = examples.sample(frac=1)
examplespos = delnegorpos(examples, 0)     # 0 - убриает отрицательные, 1 - положительные
examplesneg = delnegorpos(examples, 1)     # 0 - убриает отрицательные, 1 - положительные

examplesneg = np.array(examplesneg)
examplespos = np.array(examplespos)
balanced = []
for i in range(len(examplespos)):
    balanced.append(examplesneg[i])
    balanced.append(examplespos[i])

examples = np.array(balanced)
"""

examples = np.array(examples)

# нормализация от 0 до 1
examples = normbymax(examples)

"""
features = ['saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'consnewareas', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'funds', 'hospitals', 'beforeschool', 'factoriescap']
            
            features = ['saldo', 'foodseats', 'sportsvenue', 'servicesnum', 'museums', 'parks', 'theatres',
            'library', 'cultureorg', 'musartschool']
            
            features = ['saldo', 'foodseats', 'sportsvenue', 'servicesnum', 'theatres', 'library', 'cultureorg', 'musartschool']
            
            features = ['saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'consnewareas', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'funds', 'hospitals', 'beforeschool', 'factoriescap']
            
            features = ['saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']
"""

features = ['reg', 'interreg', 'internat', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
            'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest',
            'agrprod', 'hospitals', 'beforeschool']


examples = pd.DataFrame(examples, columns=features)

examples.to_csv("superdataset-24 3mig 2Y.csv", index=False)

print('Done')
