import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

rawdata = pd.read_csv("citiesdataset 10-21 (+y).csv")
#rawdata = np.array(rawdata)

#сортировка по именам и году
rawdata = rawdata.sort_values(by=['name', 'year'])

examples = []

# формирование датасета с социально-экономическими показателями предыдущего года
# но миграционным сальдо следующего
for i in range(len(rawdata) - 1):
    if rawdata.iloc[i, 0] == rawdata.iloc[i + 1, 0]:
        rawdata.iloc[i, 20] = rawdata.iloc[i + 1, 20]
        examples.append(rawdata.iloc[i])

"""
# вычисление среднего для каждого признака
avg = []
tmpavg = 0
count = 0
for k in range(2, len(examples[1])):
    for i in range(len(examples)):
        if examples[i, k] == examples[i, k]:  # проверка NaN
            try:
                tmpavg+= float(examples[i, k])
                count+=1
            except ValueError:
                tmpavg+=0
        else:
            tmpavg+=0
    avg.append(tmpavg / count)
    tmpavg = 0
    count = 0

# перевод из текста в число (заменить средним при невозможности конвертации)
i = 0
while i < len(rawdata):
    for j in range(len(rawdata[1])):
        if rawdata[i, j] == rawdata[i, j]:  # проверка NaN
            try:
                rawdata[i, j] = float(rawdata[i, j])
            except ValueError:
                rawdata[i, j] = avg[j]
                i -= 1
                break
        else:
            rawdata[i, j] = avg[j]
            i -= 1
            break
    i += 1

#удаляем из датасета Москву и Питер
i = 0
while i < len(rawdata):
    if rawdata[i, 0] == 'Москва' or rawdata[i, 0] == 'Санкт-Петербург':
        rawdata = np.delete(rawdata, i, 0)
        i-=1
    else:
        i+=1
"""

examples = pd.DataFrame(examples)
examples.to_csv("citiesdataset 10-21 (NY).csv", index=False)

print('Done')