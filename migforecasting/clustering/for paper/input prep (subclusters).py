import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


subclusters = pd.read_excel("medians of subclusters.xlsx")
data = pd.read_excel("input to change.xlsx")

changes = []
tmp = []
for i in range(len(data)):
    for j in range((len(subclusters))):
        if data.iloc[i]['clust'] == subclusters.iloc[j]['clust']:
            for k in range(data.shape[1]):
                if data.iloc[i][k] < subclusters.iloc[j][k]:
                    tmp.append(subclusters.iloc[j][k])
                else:
                    tmp.append(data.iloc[i][k])
            changes.append(tmp)
            tmp = []
            break

features = data.columns
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

changes.to_excel("input with changes.xlsx", index=False)