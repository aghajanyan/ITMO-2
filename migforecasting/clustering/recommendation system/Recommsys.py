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
                tmp.append(float(medians.iloc[0][col] / input.iloc[0][col]))
            break

changes.append(tmp)
features = list(input.iloc[:, 6:].columns)
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

plt.title("Сбалансированный вектор развития "+ input.iloc[0, 2] +" относительно лучшей группы мун. образований")
plt.xlabel('Во сколько раз необходимо улучшить')
plt.ylabel('Социально-экономические индикаторы')
plt.barh(changes.columns, changes.iloc[0], label=input.iloc[0, 5])
plt.show()

