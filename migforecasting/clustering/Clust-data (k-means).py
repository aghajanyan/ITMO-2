import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import label_binarize, MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import shap

# maxsaldo = 687  # 24 (2022-clust)
maxsaldo = 1015  # 24 (alltime-clust)

# popmax = 102913

k = 6  # кол-во кластеров

data = pd.read_csv("superdataset-24 alltime-clust (oktmo+name).csv")

# вывод графика с поселениями на карте согласно их реальным координатам
def townsmap():
    best = pd.read_csv("coordinates/coordinates-best.csv")
    worst = pd.read_csv("coordinates/coordinates-worst.csv")
    big = pd.read_csv("coordinates/coordinates-big.csv")

    same = pd.merge(best, worst, on=['name'], how='left')
    same = same.dropna()

    for index, row in best.iterrows():
        for i in range(len(same)):
            if row['name'] == same.iloc[i]['name']:
                best = best.drop(index)

    for index, row in worst.iterrows():
        for i in range(len(same)):
            if row['name'] == same.iloc[i]['name']:
                worst = worst.drop(index)

    plt.scatter(best['lon'], best['lat'], label="Best", marker='o', color="green")
    plt.scatter(worst['lon'], worst['lat'], label="Worst", marker='o', color="red")
    plt.scatter(big['lon'], big['lat'], label="Big", marker='o', color="black")
    #plt.scatter(same['lon_y'], same['lat_y'], label="Uncertain", marker='o', color="orange")

    plt.legend()
    plt.show()


# анализ движения поселений между кластерами (отслеживание факторного изменения)
def movementanalyzer(data):
    notsame = 0
    onetimechange = []
    changeindex = 0
    data = data.sort_values(by=['oktmo', 'year'])

    data = data[data.columns.drop('x')]
    data = data[data.columns.drop('y')]

    for i in range(len(data) - 1):  # если кластер i и i+1 не совпдатает, тогда notsame +1
        if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
            if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
                notsame += 1
                if notsame - 1 == 0:
                    changeindex = i + 1
        else:
            # перешёл в другой кластер и вернулся обратно
            if notsame == 2 and data.iloc[changeindex - 1]['clust'] == data.iloc[changeindex + 1]['clust']:
                onetimechange.append(data.iloc[changeindex - 1])
                onetimechange.append(data.iloc[changeindex])
                onetimechange.append(data.iloc[changeindex + 1])

            changeindex = 0
            notsame = 0

    onetimechange = np.array(onetimechange)
    vector = []
    tmp = []
    # в процессе разработки !!!
    for i in range(onetimechange.shape[0] - 2):
        if onetimechange[i, 0] == onetimechange[i + 1, 0] == onetimechange[i + 2, 0]:
            vector.append(np.append(onetimechange[i, :4], [0] * 16))
            for j in range(4, onetimechange.shape[1]):
                print(onetimechange[i, j])


# анализ кластеров
def analyzer(data, clusts):
    minyear = data['year'].min()
    maxyear = data['year'].max()
    data = data.sort_values(by=['oktmo', 'year'])
    notsame = 0
    solid = 0
    unic_towns = len(pd.unique(data['oktmo']))

    # вычисление кол-ва поселений, которые не меняют свой кластер за временной промежуток
    for i in range(len(data) - 1):
        if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
            if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
                notsame += 1
        else:
            if notsame == 0:
                solid += 1
            else:
                notsame = 0

    # вычисление кол-во данных за конкретный год в кластере
    cluster_year = []
    tmp = []
    for k in range(len(clusts)):
        tmp = []
        for y in range(minyear, maxyear + 1):
            tmp.append(len(clusts[k][clusts[k]['year'] == y]))
        cluster_year.append(tmp)

    cluster_year = np.array(cluster_year)
    cluster_year = pd.DataFrame(cluster_year)
    cluster_year.to_excel("years in clusters.xlsx")

    print('ok')


# доля поселений с отрицательным сальдо в класетер
def getnegative(clusts):
    negprop = []
    for i in range(len(clusts)):
        negprop.append(len(clusts[i][clusts[i]['saldo'] < 0]) / len(clusts[i]))
        plt.bar(i, negprop[i], width=0.3, label="Cluster " + str(i) + "")

    plt.title("Доля поселений с отрицательным сальдо в кластере")
    plt.legend()
    plt.xlabel('Номер кластера')
    plt.ylabel('Процент')
    plt.show()

    # выгрузка уникальных поселений из наиболее и наименее отрицательного кластера (согласно сальдо)
    minindex = negprop.index(min(negprop))
    maxindex = negprop.index(max(negprop))

    bestcities = clusts[minindex]['name'].unique()
    bestcities = np.sort(bestcities)
    bestcities = bestcities.reshape(-1, 1)

    bestcities = pd.DataFrame(bestcities)
    bestcities.to_csv("bestcities.csv", index=False)

    worstcities = clusts[maxindex]['name'].unique()
    worstcities = np.sort(worstcities)
    worstcities = worstcities.reshape(-1, 1)

    worstcities = pd.DataFrame(worstcities)
    worstcities.to_csv("worstcities.csv", index=False)


# медианное значение сальдо в кластере
def getmedian(data2):
    sns.boxplot(x='clust', y='saldo', data=data2)

    plt.title("Медианное значение сальдо в кластере")
    plt.xlabel('Номер кластера')
    plt.ylabel('Сальдо')
    plt.show()


# оценка значимости через классификатор
def findsignif(data2):
    y = data2['clust']
    y = y / (k - 1)  # нормализация

    data2 = data2[data2.columns.drop('clust')]

    class_model = RandomForestRegressor(n_estimators=100, random_state=0)
    class_model.fit(data2, y)

    explainer = shap.TreeExplainer(class_model)
    shap_values = explainer(data2).values

    shap.summary_plot(shap_values, data2)


data = data.sample(frac=1)  # перетасовка

# модель кластеризации
clust_model = KMeans(n_clusters=k, random_state=None, n_init='auto')
clust_model.fit(data.iloc[:, 3:])

# добавляем к данным столбец с номером кластера
data['clust'] = clust_model.labels_

cols = ['oktmo', 'year', 'name', 'clust', 'saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
        'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
        'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']

data = data[cols]

# data.to_csv("data-cities.csv", index=False)

# трансформация в 2D методом компонент
pca = PCA(2)
pca2 = pca.fit_transform(data.iloc[:, 3:])
data['x'] = pca2[:, 0]
data['y'] = pca2[:, 1]

# разделяем кластеры по независимым массивам (массив массивов)
clusts = []
for i in range(k):
    clusts.append(data[data['clust'] == i])

# анализ и вывод результатов

movementanalyzer(data)

#analyzer(data, clusts)

#getmedian(data)

#getnegative(clusts)

# findsignif(data)

x = [1, 2, 3, 1, 2, 3]
y = [2, 3, 2, -2, -3, -2]

for i in range(k):
    clusts[i]['x'] = x[i]
    clusts[i]['y'] = y[i]
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "", marker='o', s=160)

data = data.sort_values(by=['oktmo', 'year'])

# вычисление количества пермещений в конкретном направлении между кластерами
relation = {}
for i in range(int(len(data) - 1)):
    if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
        if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
            try:
                relation["" + str(int(data.iloc[i]['clust'])) + "-" + str(int(data.iloc[i + 1]['clust'])) + ""] += 1
            except KeyError:
                relation["" + str(int(data.iloc[i]['clust'])) + "-" + str(int(data.iloc[i + 1]['clust'])) + ""] = 1

relation = dict(sorted(relation.items(), key=lambda item: item[1], reverse=True))

for key in relation:
    cor = key.split('-')
    plt.plot((x[int(cor[0])], x[int(cor[1])]), (y[int(cor[0])], y[int(cor[1])]),
             linewidth=(relation[key] * 0.03), color="black")

"""
count = 0
for i in range(int(len(data) - 1)):
    if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
        if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
            #plt.plot(data.iloc[i:i+2]['x'], data.iloc[i:i+2]['y'], color='grey')
            plt.plot((x[int(data.iloc[i]['clust'])], x[int(data.iloc[i + 1]['clust'])]),
                     (y[int(data.iloc[i]['clust'])], y[int(data.iloc[i + 1]['clust'])]), color='grey')
            count+=1
"""

plt.legend()
plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print('done')
