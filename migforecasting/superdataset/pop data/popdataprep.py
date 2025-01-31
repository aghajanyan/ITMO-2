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
    for row in reader:
        #if row[4] == 'Всего':
        data.append(np.array(row))

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

# убрать когорты и оставить только данные за конкретные года (0, 1, ..., 69)
data = data[data['vozr'].str.isdigit()]

data = data.astype({"vozr": int})
data = data.sort_values(by=['oktmo', 'vozr'])

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
