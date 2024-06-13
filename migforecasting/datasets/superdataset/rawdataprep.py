import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

#rawdata = pd.read_csv("data_Y48112001_112_v20240402.csv", encoding='cp1251')
data = []
"""
with open('r1.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for i in range(300):
        row = next(reader)
        data.append(np.array(row))
"""
# для миграции
with open('in2022.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[4] == 'Всего' and row[5] == 'Всего' and row[6] == 'Миграция, всего':
            data.append(np.array(row))

data = np.array(data)

prepdata = []
tmp = []
b = 0
while b < len(data):
    tmp.append(data[b, 11])
    tmp.append(data[b, 14])
    tmp.append(data[b, 15])
    prepdata.append(np.array(tmp))
    tmp.clear()
    b+=1

titles = {'name', 'year', 'inflow'}

prepdata = pd.DataFrame(prepdata, columns=titles)

prepdata.to_csv("inflow 2022 (allmun).csv", index=False)

print("hello")