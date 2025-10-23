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

inputdata2 = pd.read_csv('datasets/superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict, no output).csv')
inputdata = pd.read_csv('datasets/agedata.csv')
output = pd.read_excel('Conflict assessment (top300) 21 neworder formodel norm11.xlsx')

#output = output[output['sum'] != 0]
"""
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
"""

output = output.sort_values(by=['oktmo', 'year'])

# преобразование социального риска в шкалу от 0 до 1
output['sum'] = output['sum'] / output['sum'].max()

"""
# преобразование половозрастной структуры без учёта гендера (средняя доля)
avgage = []
tmp = []
a = 0
while a < len(inputdata):
    tmp.append(int(inputdata.iloc[a, 0]))
    tmp.append(int(inputdata.iloc[a, 1]))
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

# преобразование половозрастной структуры в одну строку
male = inputdata[inputdata['gender'] == 'male']
female = inputdata[inputdata['gender'] == 'female']

male  = male.drop(columns=['name', 'gender'])
female  = female.drop(columns=['name', 'gender'])

avgage = pd.merge(male, female, how='inner', on=['oktmo', 'year'], suffixes=['_male', '_female'])

# соц-эко + демография
inputdata2  = inputdata2.drop(columns=['name'])
final = pd.merge(inputdata2, avgage, how='inner', on=['oktmo', 'year'])


"""
# преобразование в шкалу от 0 до 1
for col in avgage.columns:
    if col != 'oktmo' and col != 'name':
        avgage[col] = avgage[col].astype(float)
        if col != 'year':
            avgage[col] = avgage[col] / avgage[col].max()
"""

inputdata = final

inputdata['risk'] = list(output['sum'])

inputdata['year'] = inputdata['year'].astype(float)
inputdata['oktmo'] = inputdata['oktmo'].astype(float)

# подготовка входного и выходного результата для модели
# совмещение факторов теущего года с социальным риском следующего
examples = []
for i in range(len(inputdata) - 1):
    if inputdata.iloc[i]['oktmo'] == inputdata.iloc[i + 1]['oktmo']:
        if inputdata.iloc[i]['year'] + 1 == inputdata.iloc[i + 1]['year']:
            #inputdata.iloc[i, inputdata.shape[1] - 1] = inputdata.iloc[i + 1, inputdata.shape[1] - 1]
            inputdata.loc[i, 'risk'] = inputdata.loc[i + 1, 'risk']
            examples.append(inputdata.iloc[i])


examples = np.array(examples)
features = inputdata.columns
examples = pd.DataFrame(examples, columns=features)

#examples = examples[examples['risk'] != 0]

examples = examples.drop(columns=['oktmo', 'year'])
examples.to_csv('agerow-superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict-21, top300, formodel-2, norm11).csv', index=False)
