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

#maxsaldo = 687  # 24 (2022-clust)
maxsaldo = 1015  # 24 (alltime-clust)

#popmax = 102913

k = 6  # кол-во кластеров

data = pd.read_csv("superdataset-24 alltime-clust (oktmo).csv")


# анализ кластеров
def analyzer(data, clusts):
    minyear = data['year'].min()
    maxyear = data['year'].max()

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

    data2 = data2.sort_values(by=['oktmo', 'year'])
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
    y = y / (k - 1)     # нормализация

    data2 = data2[data2.columns.drop('clust')]

    class_model = RandomForestRegressor(n_estimators=100, random_state=0)
    class_model.fit(data2, y)

    explainer = shap.TreeExplainer(class_model)
    shap_values = explainer(data2).values

    shap.summary_plot(shap_values, data2)


data = data.sample(frac=1)  # перетасовка

# модель кластеризации
clust_model = KMeans(n_clusters=k, random_state=None, n_init='auto')
clust_model.fit(data.iloc[:, 2:])

# добавляем к данным столбец с номером кластера
data['clust'] = clust_model.labels_

cols = ['oktmo', 'year', 'clust', 'saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
        'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
        'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool']

data = data[cols]

# разделяем кластеры по независимым массивам (массив массивов)
clusts = []
for i in range(k):
    clusts.append(data[data['clust'] == i])

analyzer(data, clusts)

getnegative(clusts)

#findsignif(data)

# трансформация в 2D методом компонент
pca = PCA(2)
pca2 = pca.fit_transform(data)
data['x'] = pca2[:, 0]
data['y'] = pca2[:, 1]

for i in range(k):
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "")

plt.legend()
plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print('done')
