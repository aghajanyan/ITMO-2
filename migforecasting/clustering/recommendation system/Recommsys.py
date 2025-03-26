import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

# факторы для нормирования
features = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']


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
            infnorm = 1 - (inflation.iloc[0]['inf'] / 100)
            tonorm.iloc[k, index] = tonorm.iloc[k][col] * infnorm

    return tonorm


# нормирование факторов на душу населения
def normpersoul(tonorm):
    for k in range(len(tonorm)):
        for col in features:
            index = tonorm.columns.get_loc(col)
            tonorm.iloc[k, index] = float(tonorm.iloc[k][col] / tonorm.iloc[k]['popsize'])


# тестирование новой рекомендательной системы (profile-free)
def recommsys2():
    medians = pd.read_csv('medians all.csv')
    normpersoul(medians)

    mse = []
    for i in range(0, len(medians)):
        mse.append(mean_squared_error(medians.iloc[5][3:17], medians.iloc[i][3:17]))

    mse = pd.DataFrame(mse)
    mse.to_excel('mse.xlsx', index=False)

    medians['mse'] = mse
    print('done')


#recommsys2()
inputdata = pd.read_excel("input.xlsx")
normpersoul(inputdata)

changes = []
tmp = []
filename = ''
# выброр медиан кластеров согласно уровню МО
if inputdata.iloc[0]['type'] == 'all':
    filename = 'medians all.csv'
else:
    filename = 'medians only mundist.csv'

medians = pd.read_csv(filename)
normpersoul(medians)

# вычисление разницы входа от медиан лучшего кластера
for i in range(len(medians)):
    if inputdata.iloc[0]['profile'] == medians.iloc[i]['profile']:
        for col in inputdata.iloc[:, 8:]:
            #tmp.append(float(((medians.iloc[i][col] / inputdata.iloc[0][col])-1)*100))
            tmp.append(float(medians.iloc[i][col] / inputdata.iloc[0][col]))

        changes.append(tmp)
        tmp = []
        break

features = list(inputdata.iloc[:, 8:].columns)
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

for a in changes.columns:
    if changes.iloc[0][a] < 1:
        changes = changes[changes.columns.drop(a)]

changes = changes.transpose()

ax = changes.plot.barh()
ax.set_title("Сбалансированный вектор развития "+ inputdata.iloc[0]['name'] + " относительно лучшей группы мун. образований")
ax.set_xlabel('Во сколько раз необходимо улучшить')
ax.set_ylabel('Социально-экономические индикаторы')
plt.xlim(1, changes[0].max())
plt.legend(inputdata['profile'])
plt.show()

