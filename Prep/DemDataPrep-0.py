import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DemForecasting:
    def ExtAvgRiseAbs(olddata, cycles):
        inc = 0
        for i in range(1, len(olddata)):
            inc += olddata[i] - olddata[i - 1]

        inc = inc / len(olddata)
        for i in range(cycles):
            olddata.append(olddata[len(olddata) - 1] + inc)

data = pd.read_excel("data0.xlsx", sheet_name=3)

districtdata = []
year = []
for j in range(1, data.shape[1]):
    districtdata.append(data.iloc[1, j])
    year.append(data.iloc[0, j])

n = 5
DemForecasting.ExtAvgRiseAbs(districtdata, n)

for i in range(n):
    year.append(year[len(year) - 1] + 1)

plt.plot(year, districtdata, '.', color='black', markersize=7)
plt.plot(year[:len(districtdata) - n], districtdata[:len(districtdata) - n], color='blue', label='РОССТАТ')
plt.plot(year[len(districtdata) - n - 1:], districtdata[len(districtdata) - n - 1:], color='red', label='Прогноз')
plt.legend(loc='upper left')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(data.columns[0])
plt.show()
