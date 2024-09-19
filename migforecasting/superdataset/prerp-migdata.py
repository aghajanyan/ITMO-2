import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

regdata = []
interregdata = []
internatdata = []
"""
with open('LO outflow.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for i in range(300):
        row = next(reader)
        data.append(np.array(row))
"""
# для миграции
with open('inflow 2021.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[4] == 'Всего' and row[5] == 'Всего' and row[6] == 'Внутрирегиональная':
            regdata.append(np.array(row))
        if row[4] == 'Всего' and row[5] == 'Всего' and row[6] == 'Межрегиональная':
            interregdata.append(np.array(row))
        if row[4] == 'Всего' and row[5] == 'Всего' and row[6] == 'Международная':
            internatdata.append(np.array(row))


regdata = np.array(regdata)
interregdata = np.array(interregdata)
internatdata = np.array(internatdata)

reg = []
tmp = []
b = 0
while b < len(regdata):
    tmp.append(regdata[b, 12])
    tmp.append(regdata[b, 11])
    tmp.append(regdata[b, 14])
    tmp.append(regdata[b, 15])
    reg.append(np.array(tmp))
    tmp.clear()
    b+=1

interreg = []
tmp = []
b = 0
while b < len(interregdata):
    tmp.append(interregdata[b, 12])
    tmp.append(interregdata[b, 11])
    tmp.append(interregdata[b, 14])
    tmp.append(interregdata[b, 15])
    interreg.append(np.array(tmp))
    tmp.clear()
    b+=1

internat = []
tmp = []
b = 0
while b < len(internatdata):
    tmp.append(internatdata[b, 12])
    tmp.append(internatdata[b, 11])
    tmp.append(internatdata[b, 14])
    tmp.append(internatdata[b, 15])
    internat.append(np.array(tmp))
    tmp.clear()
    b+=1

titles = ['oktmo', 'name', 'year', 'inflow']

reg = pd.DataFrame(reg, columns=titles)
interreg = pd.DataFrame(interreg, columns=titles)
internat = pd.DataFrame(internat, columns=titles)

reg = reg.drop_duplicates()
interreg = interreg.drop_duplicates()
internat = internat.drop_duplicates()

reg.to_csv("inflow reg 2021 (allmun).csv", index=False)
interreg.to_csv("inflow interreg 2021 (allmun).csv", index=False)
internat.to_csv("inflow internat 2021 (allmun).csv", index=False)
