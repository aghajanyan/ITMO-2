import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Population:
    female = 0
    male = 0
    malemortality = 0
    femalemortality = 0
    cohortname = "null"

    def __init__(self, cohortname, female, male):
        self.cohortname = cohortname
        self.female = female
        self.male = male

    def SetMale(self, newmale):
        self.male = newmale

    def SetFemale(self, newfemale):
        self.female = newfemale

    def total(self):  # общее количество в когороте
        return self.male + self.female

    def future(self, cycle):  # выживаемость к следующему циклу
        return self.female * pow(self.femalemortality, cycle), self.male * pow(self.malemortality, cycle)


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
    def ComponentMethod(startdata, cycles):
        # инициализация
        pop = []
        i = 0
        while i < len(startdata) - 3:
            pop.append(Population(startdata[i], startdata[i + 2], startdata[i + 3]))
            i += 4

        # получение коэффициентов смертности
        mr = pd.read_excel("mr.xlsx")
        x = 0.04    # темп роста смертности для когорот 75 и старше
        for i in range(len(pop)):
            if i < mr.shape[0] - 1:
                pop[i].malemortality = mr.iloc[i + 1, 3]
                pop[i].femalemortality = mr.iloc[i + 1, 4]
            else:
                pop[i].malemortality = mr.iloc[mr.shape[0]-1, 3] - x
                pop[i].femalemortality = mr.iloc[mr.shape[0]-1, 4] - x
                x+= 0.04
        return '123'


data = pd.read_excel("data0.xlsx", sheet_name=0)

# подготовка данных для методов экстраполяции (среднрй темп роста общей численности)
districtdata = []
year = []
for j in range(1, data.shape[1]):
    districtdata.append(data.iloc[1, j])
    year.append(data.iloc[0, j])

n = 5
DemForecasting.ExtAvgRise(districtdata, n)
# DemForecasting.ExtAvgRiseAbs(districtdata, n)

# подготовка данных для метода передвижки (половозрастной столбец за 23)
fulldata23 = []
m = 3
for i in range(9, data.shape[0]):
    if data.iloc[i, data.shape[1] - 1] == data.iloc[i, data.shape[1] - 1]:
        if m == 3:
            fulldata23.append(data.iloc[i - 1, 0])
            m = 0
        fulldata23.append(data.iloc[i, data.shape[1] - 1])
        m += 1

DemForecasting.ComponentMethod(fulldata23, 1)

for i in range(n):
    year.append(year[len(year) - 1] + 1)

plt.plot(year, districtdata, '.', color='black', markersize=7)
plt.plot(year[:len(districtdata) - n], districtdata[:len(districtdata) - n], color='blue', label='РОССТАТ')
plt.plot(year[len(districtdata) - n - 1:], districtdata[len(districtdata) - n - 1:], color='red',
         label='Прогноз экстрапол.')
plt.legend(loc='upper left')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(data.columns[0])
plt.show()
