import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

"""
protests = pd.read_csv('Public view RUS.csv')
events = protests[protests['event_type'] == 'собрание']
events = events[events['subject_type'] != 'политический протест']
events = events[events['region'] != 'Москва']
"""

inputdata = pd.read_csv('superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict, no output).csv')
output = pd.read_excel('Conflict assessment (top300) 21.xlsx')

oktmo = np.unique(output['oktmo'])

# социальный риск по годам
overall = []
for i in range(9):
    overall.append(output[output['year'] == 2022 - i]['sum'].sum())

# анализ количества мун. образ. вошедших в топ близких хотя бы один раз
count = 0
for i in range(len(oktmo)):
    b = output[output['oktmo'] == oktmo[i]]['sum'].sum()
    if b == 0:
        count+=1

output = output.sort_values(by=['oktmo', 'year'])

conflictsnum = 21
# преобразование социального риска в шкалу от 0 до 1
output['sum'] = output['sum'] / conflictsnum

"""
# преобразование половозрастной структуры без учёта гендера (средняя доля)
avgage = []
tmp = []
a = 0
while a < len(inputdata):
    tmp.append(inputdata.iloc[a, 0])
    tmp.append(inputdata.iloc[a, 1])
    tmp.append(inputdata.iloc[a, 2])
    for j in range(4, inputdata.shape[1]):
        tmp.append((inputdata.iloc[a, j] + inputdata.iloc[a + 1, j]) / 2)
    avgage.append(np.array(tmp))
    tmp = []
    a+=2

inputdata = inputdata.drop(columns=['gender'])
avgage = np.array(avgage)
features = inputdata.columns
avgage = pd.DataFrame(avgage, columns=features)

"""
"""
# преобразование половозрастной структуры в одну строку
male = inputdata[inputdata['gender'] == 'male']
female = inputdata[inputdata['gender'] == 'female']

male  = male .drop(columns=['name', 'gender'])
female  = female .drop(columns=['name', 'gender'])

# x - мужчины, y - женщины
avgage = pd.merge(male, female, how='inner', on=['oktmo', 'year'])


# преобразование в шкалу от 0 до 1
for col in avgage.columns:
    if col != 'oktmo' and col != 'name':
        avgage[col] = avgage[col].astype(float)
        if col != 'year':
            avgage[col] = avgage[col] / avgage[col].max()
"""

#inputdata = avgage

inputdata['risk'] = list(output['sum'])

# подготовка входного и выходного результата для модели
# совмещение факторов теущего года с социальным риском следующего
examples = []
for i in range(len(inputdata) - 1):
    if inputdata.iloc[i]['oktmo'] == inputdata.iloc[i + 1]['oktmo']:
        if inputdata.iloc[i]['year'] + 1 == inputdata.iloc[i + 1]['year']:
            inputdata.iloc[i, inputdata.shape[1] - 1] = inputdata.iloc[i + 1, inputdata.shape[1] - 1]
            examples.append(inputdata.iloc[i])


examples = np.array(examples)
features = inputdata.columns
examples = pd.DataFrame(examples, columns=features)

examples = examples.drop(columns=['oktmo', 'year', 'name'])
examples.to_csv('superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21).csv', index=False)
