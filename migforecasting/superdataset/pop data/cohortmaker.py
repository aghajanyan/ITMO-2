import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


for y in range(2011, 2024):
    data = pd.read_csv("agestruct (per age)/agestruct "+ str(y) +".csv")

    # когорты и будущие столбцы в таблице
    cols = ['0-4', '5-9', '10-14', '15-19', '20-24',
            '25-29', '30-34', '35-39', '40-44', '45-49',
            '50-54', '55-59', '60-64', '65-69']

    cohort = []
    tmp = []
    n = 0   # кол-во хюмансов в когорте
    # преобразование данных за год в когорту
    for i in range(10):
        for a in cols:
            a = a.split('-')
            n = 0
            for j in range(int(a[0]), int(a[1]) + 1):
                n += data.iloc[i][''+str(j)+'']
            tmp.append(n)
        cohort.append(np.array(tmp))
        tmp.clear()

    cohort = np.array(cohort)
    cohort = pd.DataFrame(cohort, columns=cols)
    cohort[['oktmo', 'name', 'gender', 'year']] = data[['oktmo', 'name', 'gender', 'year']]

    # порядок столбцов в таблице
    cols = ['oktmo', 'name', 'gender', 'year'] + cols
    cohort = cohort[cols]

    cohort.to_csv("agestruct (per cohort)/agestruct "+ str(y) +" (cohort).csv", index=False)
    print('Итерация № ', y)

print('Done')
