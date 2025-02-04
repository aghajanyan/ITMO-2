import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

data = []
"""
with open('LO outflow.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for i in range(300):
        row = next(reader)
        data.append(np.array(row))
"""
# для миграции
with open('pop23.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for i in range(1300):
        row = next(reader)
        if row[4] == 'Всего' or i == 0:
            data.append(np.array(row))
    #for row in reader:
        #if row[4] == 'Всего':
            #data.append(np.array(row))

data = np.array(data)

data = pd.DataFrame(data, columns=data[0])

data = data.sort_values(by=['oktmo', 'vozr'])
"""
newdata = []
for i in range(len(data)):
    if len(list(data.iloc[i]['vozr'])) < 3:
        newdata.append(data.iloc[i])

newdata = np.array(newdata)
newdata = pd.DataFrame(newdata, columns=data[0])

for index, row in data.iterrows():
    try:
        tmp = int(row['vozr'])
    except ValueError:
        data = data.drop(index)

"""

cols = list(['oktmo', 'municipality', 'year', 'indicator_value']) + list(range(0, 70))


# убрать когорты и оставить только данные за конкретные года (0, 1, ..., 69)
data = data[data['vozr'].str.isdigit()]

data = data.astype({"vozr": int})
data = data.sort_values(by=['oktmo', 'vozr'])

newdata = data[['oktmo', 'municipality', 'year', 'vozr', 'indicator_value']]

tmp = []
final = []
i = 0
while i < len(newdata) - 1:
    if newdata.iloc[i]['oktmo'] != newdata.iloc[i+1]['oktmo']:
        tmp.append(newdata.iloc[i]['indicator_value'])
        final.append(np.array(tmp))
        tmp.clear()
    else:
        if newdata.iloc[i]['vozr'] != newdata.iloc[i+1]['vozr']:
            if len(tmp) != 0:
                tmp.append(newdata.iloc[i]['indicator_value'])
            else:
                tmp.append(newdata.iloc[i]['oktmo'])
                tmp.append(newdata.iloc[i]['indicator_value'])
        else:
            if newdata.iloc[i]['indicator_value'] < newdata.iloc[i+1]['indicator_value']:
                tmp.append(newdata.iloc[i]['indicator_value'])
                i+=1
            else:
                tmp.append(newdata.iloc[i+1]['indicator_value'])
                i+=1
    i += 1




prepdata = []
tmp = []
b = 0
oktmo = data[0, 11]
name = data[0, 9]
year = data[0, 13]
total = 0
while b < len(data):
    tmp.append(data[b, 12])
    tmp.append(data[b, 11])
    tmp.append(data[b, 14])
    tmp.append(data[b, 15])
    prepdata.append(np.array(tmp))
    tmp.clear()
    b+=1

titles = ['oktmo', 'name', 'year', 'inflow']

prepdata = pd.DataFrame(prepdata, columns=titles)

prepdata = prepdata.drop_duplicates()

prepdata.to_csv("inflow reg 2021.csv", index=False)
