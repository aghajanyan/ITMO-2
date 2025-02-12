import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


for y in range(2011, 2024):
    data = pd.read_csv("agestruct (per cohort)/agestruct "+ str(y) +" (cohort).csv")

    cols = ['0-4', '5-9', '10-14', '15-19', '20-24',
            '25-29', '30-34', '35-39', '40-44', '45-49',
            '50-54', '55-59', '60-64', '65-69']

    # общее количество людей (сумма всех когорт)
    data['total'] = data.iloc[:, 4: data.shape[1]].sum(axis=1)

    allprop = []
    tmp = []
    # вычисление доли в когорте от общего числа людей
    for i in range(len(data)):
        for j in cols:
            prop = data.iloc[i][j] / data.iloc[i]['total']
            tmp.append(np.float16(prop))
        allprop.append(np.array(tmp))
        tmp.clear()

    # создание таблицы
    allprop = np.array(allprop)
    allprop = pd.DataFrame(allprop, columns=cols)
    allprop[['oktmo', 'name', 'gender', 'year']] = data[['oktmo', 'name', 'gender', 'year']]

    # порядок столбцов в таблице
    cols = ['oktmo', 'name', 'gender', 'year'] + cols
    allprop = allprop[cols]

    allprop.to_csv("agestruct (prop)/agestruct "+ str(y) +" (prop).csv", index=False)
    print('Итерация № ', y)

print('Done')
