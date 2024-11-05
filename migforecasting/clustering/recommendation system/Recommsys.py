import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def normpersoul(tonorm):
    for k in range(len(tonorm)):
        for col in tonorm.iloc[:, 7:]:
            if col != 'avgsalary' and col != 'livarea' and col != 'roadslen' and col != 'hospitals':
                for m in range(len(tonorm)):
                    #tonorm = tonorm.astype({col: float})
                    x = float(tonorm.iloc[m][col] / tonorm.iloc[m]['popsize'])
                    tonorm.iloc[0, 7] = x

medians = pd.read_csv("medians.csv")

input = pd.read_excel("input.xlsx")

normpersoul(input)

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
ax.set_title("Сбалансированный вектор развития "+ input.iloc[0]['name'] +" относительно лучшей группы мун. образований")
ax.set_xlabel('Во сколько раз необходимо улучшить')
ax.set_ylabel('Социально-экономические индикаторы')
plt.legend(input['profile'])
plt.show()

