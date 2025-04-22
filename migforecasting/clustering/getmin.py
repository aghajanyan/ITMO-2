import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_excel("Clusters.xlsx", sheet_name=None)

tmp = []
mindata = []
for cluster in data:
    positive = data[cluster][data[cluster]['saldo'] > 0]
    tmp.append(cluster)
    for col in positive.columns[5:]:
        tmp.append(positive[col].min())
    mindata.append(tmp)
    tmp = []

features = list(data['Cluster 0'].columns[5:])
features.insert(0, 'clust')
mindata = np.array(mindata)
mindata = pd.DataFrame(mindata, columns=features)

print('ok')