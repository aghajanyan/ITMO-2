import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


medians = pd.read_csv("medians.csv")

input = pd.read_excel("input.xlsx")

changes = []
tmp = []
for k in range(len(input)):
    for i in range(len(medians)):
        if input.iloc[k]['profile'] == medians.iloc[i]['profile']:
            for col in input.iloc[:, 6:]:
                tmp.append(float(medians.iloc[i][col] / input.iloc[k][col]))

            changes.append(tmp)
            tmp = []
            break

features = list(input.iloc[:, 6:].columns)
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

changes = changes.transpose()

ax = changes.plot.barh()
ax.set_title("Сбалансированный вектор развития "+ input.iloc[0, 2] +" относительно лучшей группы мун. образований")
ax.set_xlabel('Во сколько раз необходимо улучшить')
ax.set_ylabel('Социально-экономические индикаторы')
plt.legend(input['profile'])
plt.show()

