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

gender = 'Male'

# загрузка и первичная обработка входного файла файла
with open('pop23.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    row0 = next(reader)
    data.append(np.array(row0))
    if gender == 'Male':
        for i in range(1300):
            row = next(reader)
        #for row in reader:
            if row[4] == 'Всего':
                data.append(np.array(row))
    else:
        for row in reader:
            if row[4] == 'Женщины':
                data.append(np.array(row))

data = np.array(data)

data = pd.DataFrame(data, columns=data[0])
data = data.drop(0)

data = data.sort_values(by=['oktmo', 'vozr'])

# убрать когорты и оставить только данные за конкретные года (0, 1, ..., 69)
data = data[data['vozr'].str.isdigit()]

# преобразование типов и сортировка
data = data.astype({"vozr": int})
data = data.astype({"indicator_value": float})
data = data.astype({"indicator_value": int})
data = data.sort_values(by=['oktmo', 'vozr'])

# оставляем только нужное
newdata = data[['oktmo', 'municipality', 'year', 'vozr', 'indicator_value']]
newdata['gender'] = 'Male'

# иногда один и тот же возраст повторяется, при этом данные по количеству отличаются!! (удаляем дубликаты)
newdata = newdata.drop_duplicates(subset=['oktmo', 'vozr'], keep='last')

# транспонирование с целью "номер столбца = возраст"
final = newdata.pivot(index=['oktmo'], columns='vozr', values='indicator_value')

# октмо - отдельный столбец, индекс от 0 до n
final['oktmo'] = final.index
final = final.reset_index(drop=True)

# подготовка для мёрджа названий, года и гендера
newdata = newdata[newdata.columns.drop(['vozr', 'indicator_value'])]
newdata = newdata.drop_duplicates()

final = final.merge(newdata, on='oktmo', how='left')

final = np.array(final)
cols = list(['oktmo', 'name', 'year', 'gender']) + list(range(0, 70))
final = pd.DataFrame(final, columns=cols)

final.to_csv("agestruct "+gender+" 2023.csv", index=False)


"""
pivot в ручном режиме (работает очень, очень долго)
tmp = []
final = []
i = 0
age = 0
badoktmo = []
while i < len(newdata):
    if i != len(newdata) - 1:
        if newdata.iloc[i]['oktmo'] != newdata.iloc[i+1]['oktmo']:
            tmp.append(newdata.iloc[i]['indicator_value'])
            final.append(np.array(tmp))
            tmp.clear()
            age = -1
        else:
            if newdata.iloc[i]['vozr'] != newdata.iloc[i+1]['vozr']:
                if len(tmp) != 0:
                    tmp.append(newdata.iloc[i]['indicator_value'])
                    if age != newdata.iloc[i]['vozr']:
                        print('error')
                else:
                    tmp.append(newdata.iloc[i]['oktmo'])
                    tmp.append(newdata.iloc[i]['municipality'])
                    tmp.append(newdata.iloc[i]['year'])
                    tmp.append(gender)
                    tmp.append(newdata.iloc[i]['indicator_value'])
                    if age != newdata.iloc[i]['vozr']:
                        print('error')
            else:
                if newdata.iloc[i]['indicator_value'] < newdata.iloc[i+1]['indicator_value']:
                    tmp.append(newdata.iloc[i]['indicator_value'])
                    i+=1
                    if age != newdata.iloc[i]['vozr']:
                        print('error')
                else:
                    tmp.append(newdata.iloc[i+1]['indicator_value'])
                    i+=1
                    if age != newdata.iloc[i]['vozr']:
                        print('error')
        i += 1
        age += 1
    else:
        tmp.append(newdata.iloc[i]['indicator_value'])
        final.append(np.array(tmp))
        tmp.clear()
        i += 1
"""