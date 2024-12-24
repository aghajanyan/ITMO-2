import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# факторы для нормирования
features = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
            'livestock', 'harvest', 'agrprod', 'beforeschool']


#Нормирование рублевых цен
def normbyinf(tonorm):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest',
                      'budincome', 'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    infdata = pd.read_csv("inflation14.csv")
    for k in range(len(tonorm)):
        inflation = infdata[infdata['year'] == tonorm.iloc[k]['year']]
        for col in thisrubfeatures:
            index = tonorm.columns.get_loc(col)
            tonorm.iloc[k, index] = tonorm.iloc[k][col] * (inflation.iloc[0]['inf'] / 100)

    return tonorm


# нормирование факторов на душу населения
def normpersoul(tonorm):
    for k in range(len(tonorm)):
        for col in features:
            index = tonorm.columns.get_loc(col)
            tonorm.iloc[k, index] = float(tonorm.iloc[k][col] / tonorm.iloc[k]['popsize'])


inputdata = pd.read_excel("input.xlsx")
normpersoul(inputdata)

changes = []
tmp = []
filename = ''
for k in range(len(inputdata)):
    # выброр медиан кластеров согласно уровню МО
    if inputdata.iloc[k]['type'] == 'all':
        filename = 'medians all.csv'
    else:
        filename = 'medians only mundist.csv'

    medians = pd.read_csv(filename)
    normpersoul(medians)

    # вычисление разницы входа от медиан лучшего кластера
    for i in range(len(medians)):
        if inputdata.iloc[k]['profile'] == medians.iloc[i]['profile']:
            for col in inputdata.iloc[:, 7:]:
                tmp.append(float(medians.iloc[i][col] / inputdata.iloc[k][col]))

            changes.append(tmp)
            tmp = []
            break

features = list(inputdata.iloc[:, 7:].columns)
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

changes = changes.transpose()

ax = changes.plot.barh()
ax.set_title("Сбалансированный вектор развития "+ inputdata.iloc[0]['name'] +" относительно лучшей группы мун. образований")
ax.set_xlabel('Во сколько раз необходимо улучшить')
ax.set_ylabel('Социально-экономические индикаторы')
plt.legend(inputdata['profile'])
plt.show()

