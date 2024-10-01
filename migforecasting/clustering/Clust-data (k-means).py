import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def analyzer(clusts, N):
    med = []
    minmin = []
    maxmax = []
    negprop = []
    for i in range(len(clusts)):
        med.append(clusts[i]['saldo'].median() * maxsaldo)
        minmin.append(clusts[i]['saldo'].min() * maxsaldo)
        maxmax.append(clusts[i]['saldo'].max() * maxsaldo)
        negprop.append(len(clusts[i][clusts[i]['saldo'] < 0]) / len(clusts[i]))

    plt.bar(N, negprop, width=0.3)
    plt.show()


maxsaldo = 687  # 24 (2022-clust)

data = pd.read_csv("superdataset-24 2022-clust.csv")

"""
error = []
tmper = []
N = range(2, 11)    # количество кластеров
for i in range(10):     # цикл для вычисления средней ошибки для конкретного кол-ва кластеров
    tmper = []
    data = data.sample(frac=1)  # перетасовка
    for n in N:
        # модель кластеризации и вычисление ошибки
        clust_model = KMeans(n_clusters=n, random_state=None, n_init='auto')
        clust_model.fit(data)
        tmper.append(silhouette_score(data, clust_model.labels_, metric='euclidean'))

    for j, m in enumerate(tmper):
        if i == 0:
            error.append(m)
        else:
            error[j]+=m

for i in range(len(error)):
    error[i] = error[i] / 10

plt.plot(N, error)
plt.xlabel("Number of clusters")
plt.ylabel("Error")
plt.show()
"""

k = 3
N = range(2, k + 2)

data = data.sample(frac=1)  # перетасовка
clust_model = KMeans(n_clusters=k, random_state=None, n_init='auto')
clust_model.fit(data)

data['clust'] = clust_model.labels_

# трансформация в 2D методом компонент
pca = PCA(2)
pca2 = pca.fit_transform(data)

data['x'] = pca2[:, 0]
data['y'] = pca2[:, 1]

clusts = []
for i in range(k):
    clusts.append(data[data['clust'] == i])

analyzer(clusts, N)

for i in range(k):
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "")

plt.legend()
plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print('done')
