import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DemForecasting:
    @staticmethod
    def ExtAvgRiseAbs(olddata, cycles):
        inc = 0
        for i in range(1, len(olddata)):
            inc += olddata[i] - olddata[i - 1]

        inc = inc / len(olddata)
        for i in range(cycles):
            olddata.append(olddata[len(olddata) - 1] + inc)

    @staticmethod
    def ExtAvgRise(olddata, cycles):
        inc = 0
        for i in range(1, len(olddata)):
            inc += (olddata[i] / olddata[i - 1]) - 1

        inc = inc / len(olddata)
        for i in range(cycles):
            olddata.append(olddata[len(olddata) - 1] * (inc + 1))

    @staticmethod
    def ComponentMethod(oldata, cycles):

        return "hello"


data = pd.read_excel("data0.xlsx", sheet_name=0)


#подготовка данных для методов экстраполяции (среднрй темп роста общей численности)
districtdata = []
year = []
for j in range(1, data.shape[1]):
    districtdata.append(data.iloc[1, j])
    year.append(data.iloc[0, j])

n = 5
DemForecasting.ExtAvgRise(districtdata, n)
#DemForecasting.ExtAvgRiseAbs(districtdata, n)

# подготовка данных для метода передвижки (половозрастной столбец за 23)
fulldata23 = []
for i in range(9, data.shape[0]):
    if data.iloc[i, data.shape[1] - 1] == data.iloc[i, data.shape[1] - 1]:
        fulldata23.append(data.iloc[i, data.shape[1] - 1])

DemForecasting.ComponentMethod(fulldata23, 1)

for i in range(n):
    year.append(year[len(year) - 1] + 1)

plt.plot(year, districtdata, '.', color='black', markersize=7)
plt.plot(year[:len(districtdata) - n], districtdata[:len(districtdata) - n], color='blue', label='РОССТАТ')
plt.plot(year[len(districtdata) - n - 1:], districtdata[len(districtdata) - n - 1:], color='red', label='Прогноз экстрапол.')
plt.legend(loc='upper left')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(data.columns[0])
plt.show()
