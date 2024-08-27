import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# получение исходных данных
saldo = pd.read_csv("saldo employable LO.csv")

saldo['saldo'] = saldo['saldo'].abs()
saldo = saldo.dropna()
saldo = np.array(saldo)

#total
"""
tot = []
for i in range(len(saldo)):
    if saldo[i, 3] == 'Миграция, всего':
        tot.append(saldo[i])

tot = np.array(tot)
titles = ['oktmo', 'name', 'year', 'whereabouts', 'total saldo']
tot = pd.DataFrame(tot, columns=titles)
tot.to_csv('total saldo.csv', index=False)
"""

# разбивка на отдельные массивы согласно трём направлениям
inregmas = []
outregmas = []
outmas = []
for i in range(len(saldo)):
    if saldo[i, 3] == 'Межрегиональная': #and saldo[i, 2] > 2019:
        outregmas.append(saldo[i])
    if saldo[i, 3] == 'Внутрирегиональная': #and saldo[i, 2] > 2019:
        inregmas.append(saldo[i])
    if saldo[i, 3] == 'Международная': #and saldo[i, 2] > 2019:
        outmas.append(saldo[i])

titles = ['oktmo', 'name', 'year', 'whereabouts', 'saldo']
outregmas = pd.DataFrame(outregmas, columns=titles)
inregmas = pd.DataFrame(inregmas, columns=titles)
outmas = pd.DataFrame(outmas, columns=titles)

inregavg = 0
outregavg = 0
outavg = 0

# вычисление средних долей направления миграции для Ленобласти
total = outregmas['saldo'].sum() + inregmas['saldo'].sum() + outmas['saldo'].sum()
inregavg = inregmas['saldo'].sum() / total
outregavg = outregmas['saldo'].sum() / total
outavg = outmas['saldo'].sum() / total

"""
# средняя для Ленобласти для трудоспособных (вычислить от общего числа всей миграции)
total = pd.read_csv("total saldo.csv")
total['total saldo'] = total['total saldo'].abs()
total = total[total.columns.drop('whereabouts')]
total = total[total.columns.drop('name')]

avgdistemp = outregmas.merge(total, on=['oktmo', 'year'], how='left')
avgdistemp = avgdistemp[avgdistemp.columns.drop(['name', 'whereabouts'])]
avgdistemp = avgdistemp.merge(inregmas, on=['oktmo', 'year'], how='left')
avgdistemp = avgdistemp[avgdistemp.columns.drop(['name', 'whereabouts'])]
avgdistemp = avgdistemp.merge(outmas, on=['oktmo', 'year'], how='left')
avgdistemp = avgdistemp[avgdistemp.columns.drop(['name', 'whereabouts'])]

avgdistemp = avgdistemp.dropna()

total = avgdistemp['total saldo'].sum()
inregavg = avgdistemp['saldo_y'].sum() / total
outregavg = avgdistemp['saldo_x'].sum() / total
outavg = avgdistemp['saldo'].sum() / total
"""

# формирование файла с долями направления миграции по всем поселениям (+ ленобласть)
titles = ['oktmo', 'name', 'regional', 'national', 'international']
final = []
final.append([0, 'Ленобласть', inregavg, outregavg, outavg])
final = pd.DataFrame(final, columns=titles)

# сортировка и мёрджинг для упрощения вычислений
outregmas = outregmas.sort_values(by=['oktmo', 'year'])
inregmas = inregmas.sort_values(by=['oktmo', 'year'])
outmas = outmas.sort_values(by=['oktmo', 'year'])

inregmas = inregmas[inregmas.columns.drop(['name', 'whereabouts'])]
outmas = outmas[outmas.columns.drop(['name', 'whereabouts'])]

outregmas = outregmas.merge(inregmas, on=['oktmo', 'year'], how='left')
outregmas = outregmas.merge(outmas, on=['oktmo', 'year'], how='left')

outregmas = outregmas.dropna()
outregmas = np.array(outregmas)

inregavg = 0
outregavg = 0
outavg = 0
total = 0
# вычисление долей направления для всех поселений
for i in range(len(outregmas) - 1):
    if outregmas[i, 0] == outregmas[i + 1, 0]:
        outregavg += outregmas[i, 4]
        inregavg += outregmas[i, 5]
        outavg += outregmas[i, 6]
        total += outregmas[i, 4] + outregmas[i, 5] + outregmas[i, 6]
    else:
        outregavg += outregmas[i, 4]
        inregavg += outregmas[i, 5]
        outavg += outregmas[i, 6]
        total += outregmas[i, 4] + outregmas[i, 5] + outregmas[i, 6]

        outregavg = outregavg / total
        inregavg = inregavg / total
        outavg = outavg / total
        final = final.append({'oktmo': outregmas[i, 0], 'name': outregmas[i, 1],
                              'regional': inregavg, 'national': outregavg, 'international': outavg}, ignore_index=True)
        inregavg = 0
        outregavg = 0
        outavg = 0
        total = 0

final.to_csv('migprop employable (avg alltime).csv', index=False)

print('ok')
