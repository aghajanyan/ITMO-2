import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

# основные датасеты (соц-эко и демографические данные)
data = pd.read_csv("datasets/superdataset-24 alltime-clust (IQR)-normbysoul-f.csv")
agestruct = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/pop data/agestruct prop.csv")

# данных по демографии больше, поэтому требуется их синхронизировать (эквивалентные примеры в двух сетах)
agestruct = pd.merge(data[['oktmo', 'year']], agestruct, how='inner', on=['oktmo', 'year'])
sync = agestruct[['oktmo', 'year']].drop_duplicates()
data = pd.merge(sync, data, how='inner', on=['oktmo', 'year'])

# кейсы с социальными конфликтами для оценки похожести (локальные "экстремумы")
socecoextrem = pd.read_excel('soc-eco extremums.xlsx')
ageextrem = pd.read_excel('age extremums.xlsx')

# перевод в array для более быстрого вычисления (но это не точно)
agestruct = np.array(agestruct)
ageextrem = np.array(ageextrem)
dataarr = np.array(data)
socecoextrem = np.array(socecoextrem)

# фрейм для записи финальных результатов
rankings = pd.DataFrame()
rankings['oktmo'] = data['oktmo']
rankings['name'] = data['name']
rankings['year'] = data['year']

a = 0
for k in range(len(socecoextrem)):

    # оценка подобия по социально-экономическим факторам
    sim1 = []
    tmp = 0.0
    for i in range(len(data)):
        tmp = mean_squared_error(dataarr[i, 5:], socecoextrem[k, 5:]) #popsize and saldo не используются
        sim1.append(tmp)

    # оценка подобия по половозрастной структуре
    sim2 = []
    female = 0.0
    male = 0.0
    b = 0
    # итерация через 2, поскольку отдельная строка по мужчинам и женщинам
    if k == 0:
        a = 0
    else:
        a += 2

    # проверка соответствия oktmo для соц-эко и демографии (в файлах они строго по порядку)
    if socecoextrem[k, 0] != ageextrem[a, 0] or ageextrem[a, 0] != ageextrem[a + 1, 0]:
        print('Ошибка!')

    while b < len(agestruct):
        female = mean_squared_error(agestruct[b, 4:18], ageextrem[a, 4:18])
        male = mean_squared_error(agestruct[b + 1, 4:18], ageextrem[a + 1, 4:18])
        sim2.append(female + male)
        b += 2

    # критерий подобия и сортировка
    sim3 = np.array(sim1) + (np.array(sim2) * 0.5) # значимость демографии урезается
    data['similarity'] = sim3
    data = data.sort_values(by='similarity')

    # для реализации ранжирования на основе топ-300
    risk = []
    tier1 = [1] * 100
    tier2 = [0.5] * 100
    tier3 = [0.2] * 100
    tier4 = [0] * (len(data) - 300)
    risk = tier1 + tier2 + tier3 + tier4

    # добавление оценок в финальную таблицу rankings
    data['risk'] = risk
    data = data.sort_values(by=['oktmo', 'year'])
    rankings[str(k + 1)] = data['risk']

    data = data[data.columns.drop('similarity')]
    data = data[data.columns.drop('risk')]
    print(k)

# вычисление суммарной оценки, сортировка и сохранение результата
rankings['oktmo'] = rankings['oktmo'].astype(str)
rankings['year'] = rankings['year'].astype(str)
rankings['sum'] = rankings.sum(axis=1, numeric_only=True)
rankings = rankings.sort_values(by=['sum'], ascending=False)
rankings.to_excel('Conflict assessment (top300) 21.xlsx', index=False)