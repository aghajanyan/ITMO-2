import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


subclusters = pd.read_excel("medians of subclusters.xlsx")
scdiff = pd.read_excel("scdiff.xlsx")
data = pd.read_excel("input to change.xlsx")

changes = []
tmp = []
for i in range(len(data)):
    for j in range((len(subclusters))):
        if data.iloc[i]['clust'] == subclusters.iloc[j]['clust']:
            for k in range(data.shape[1]):
                if data.iloc[i][k] < subclusters.iloc[j][k]:
                    if subclusters.iloc[j][k] / data.iloc[i][k] < 2:
                        tmp.append(subclusters.iloc[j][k]) # замена на медиану (если разница от неё меньше 100%)
                    else:
                        # увеличение на сред. процент разницы между медианами полож. и отриц. субкластера
                        tmp.append(data.iloc[i][k] + (data.iloc[i][k] * scdiff.iloc[j][k]))
                else:
                    tmp.append(data.iloc[i][k])
            changes.append(tmp)
            tmp = []
            break

features = data.columns
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

changes.to_excel("input with changes.xlsx", index=False)