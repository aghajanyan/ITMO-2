import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd



medians = pd.read_csv("medians.csv")

input = pd.read_excel("input.xlsx")

changes = []
tmp = []
for col in input.iloc[:, 5:]:
    tmp.append(float(medians.iloc[0][col] / input[col]))

changes.append(tmp)

tmp = []
for col in input.iloc[:, 5:]:
    tmp.append(float(medians.iloc[1][col] / input[col]))

changes.append(tmp)
features = list(input.iloc[:, 5:].columns)
changes = np.array(changes)
changes = pd.DataFrame(changes, columns=features)

plt.barh(changes.columns, changes.iloc[0])
plt.show()

