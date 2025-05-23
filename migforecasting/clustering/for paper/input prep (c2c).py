import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


subclusters = pd.read_excel("medians c2c.xlsx")
scdiff = pd.read_excel("scdiff c2c.xlsx")
data = pd.read_excel("input c2c.xlsx")

changes = []
tmp = []
for i in range(len(data)):
    for j in range((len(subclusters))):
        # определение наиболее близкого кластера для перехода
        tocluster = 0
        mse = mean_squared_error(data.iloc[i][1:], subclusters.iloc[0][1:])
        for m in range(1, (len(subclusters))):
           if mse > mean_squared_error(data.iloc[i][1:], subclusters.iloc[m][1:]):
               mse = mean_squared_error(data.iloc[i][1:], subclusters.iloc[m][1:])
               tocluster = m

        tmp.append(data.iloc[i][0])
        for k in range(1, data.shape[1]):
            if data.iloc[i][k] < subclusters.iloc[tocluster][k]:
                if subclusters.iloc[tocluster][k] / data.iloc[i][k] < 2:
                    tmp.append(subclusters.iloc[tocluster][k])  # замена на медиану (если разница от неё меньше 100%)
                else:
                    # увеличение на сред. процент разницы между медианами полож. и отриц. субкластера
                    tmp.append(data.iloc[i][k] + (data.iloc[i][k] * scdiff.iloc[tocluster][k]))
            else:
                tmp.append(data.iloc[i][k])
        changes.append(tmp)
        tmp = []
        break


features = data.columns
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

changes.to_excel("input with changes (c2c).xlsx", index=False)