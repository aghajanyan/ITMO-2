import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("data0.xlsx", sheet_name=0)

a = []
year = []
for j in range(1, data.shape[1]):
    a.append(data.iloc[1, j])
    year.append(data.iloc[0, j])

plt.plot(year, a, '.', color='black', markersize=7)
plt.plot(year, a, color='black')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(data.columns[0])
plt.show()