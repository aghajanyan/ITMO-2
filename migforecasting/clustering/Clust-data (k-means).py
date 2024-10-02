import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns


# анализ кластеров (медиана, доля отрицательных)
def analyzer(clusts, data2):
    data2['saldo'] = data2['saldo'] * maxsaldo
    sns.boxplot(x='clust', y='saldo', data=data2)

    plt.title("Медианное значение сальдо в кластере")
    plt.xlabel('Номер кластера')
    plt.ylabel('Сальдо')
    plt.show()

    negprop = []
    for i in range(len(clusts)):
        negprop.append(len(clusts[i][clusts[i]['saldo'] < 0]) / len(clusts[i]))
        plt.bar(i, negprop[i], width=0.3, label="Cluster " + str(i) + "")

    plt.title("Доля поселений с отрицательным сальдо в кластере")
    plt.legend()
    plt.xlabel('Номер кластера')
    plt.ylabel('Процент')
    plt.show()


maxsaldo = 687  # 24 (2022-clust) 

data = pd.read_csv("superdataset-24 2022-clust.csv")

k = 4   # кол-во кластеров

data = data.sample(frac=1)  # перетасовка

# модель кластеризации
clust_model = KMeans(n_clusters=k, random_state=None, n_init='auto')
clust_model.fit(data)

# добавляем к данным столбец с номером кластера
data['clust'] = clust_model.labels_

# трансформация в 2D методом компонент
pca = PCA(2)
pca2 = pca.fit_transform(data)
data['x'] = pca2[:, 0]
data['y'] = pca2[:, 1]

# разделяем кластеры по независимым массивам (массив массивов)
clusts = []
for i in range(k):
    clusts.append(data[data['clust'] == i])

# анализ и вывод результатов

analyzer(clusts, data)

for i in range(k):
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "")

plt.legend()
plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print('done')
