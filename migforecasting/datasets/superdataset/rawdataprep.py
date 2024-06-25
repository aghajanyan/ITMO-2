import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

#rawdata = pd.read_csv("data_Y48112001_112_v20240402.csv", encoding='cp1251')
data = []
"""
with open('LO outflow.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for i in range(300):
        row = next(reader)
        data.append(np.array(row))
"""
# для миграции
with open('fac.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[4] == 'Всего' and row[15] == 'Январь — декабрь':
            data.append(np.array(row))

data = np.array(data)

prepdata = []
tmp = []
b = 0
oktmo = data[0, 11]
name = data[0, 9]
year = data[0, 13]
total = 0
while b < len(data):
    tmp.append(data[b, 10])
    tmp.append(data[b, 9])
    tmp.append(data[b, 12])
    tmp.append(data[b, 13])
    prepdata.append(np.array(tmp))
    tmp.clear()
    b+=1
"""
while b < len(data):
    if oktmo != data[b, 11] or year != data[b, 13]:
        tmp.append(oktmo)
        tmp.append(name)
        tmp.append(year)
        tmp.append(total)
        prepdata.append(np.array(tmp))
        tmp.clear()
        total = 0
        oktmo = data[b, 11]
        name = data[b, 9]
        year = data[b, 13]
    try:
        total+= float(data[b, 14])
    except ValueError:
        total+=0
    b+=1
"""
titles = ['oktmo', 'name', 'year', 'factoriescap']

prepdata = pd.DataFrame(prepdata, columns=titles)

prepdata = prepdata.drop_duplicates()

prepdata.to_csv("f-2.csv", index=False)
