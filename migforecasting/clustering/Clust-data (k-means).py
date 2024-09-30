import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = pd.read_csv("superdataset-24 2022-clust.csv")

data = data.sample(frac=1)  # перетасовка

clust_model = KMeans(n_clusters=4, random_state=None, n_init='auto')

clust_model.fit(data)

error = silhouette_score(data, clust_model.labels_, metric='euclidean')

print(error)