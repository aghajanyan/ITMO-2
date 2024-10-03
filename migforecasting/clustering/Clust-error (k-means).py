import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("superdataset-VD alltime-clust.csv")

error = []
tmper = []
N = range(2, 11)    # количество кластеров
x = 5   # количество повторных циклов
for i in range(x):     # цикл для вычисления средней ошибки для конкретного кол-ва кластеров
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
    error[i] = error[i] / x

plt.plot(N, error)
plt.xlabel("Number of clusters")
plt.ylabel("Error")
plt.show()
