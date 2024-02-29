import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Population:
    female = 0  # количество в когорте
    male = 0
    malemortality = 0  # возрастной коэф. годовой смертности
    femalemortality = 0
    cohortname = "null"
    birthrate = 0  # возрастной коэф. рождаемости
    migrate = 0  # возрастной коэф. прибывших/выбывших

    def __init__(self, cohortname, female, male):
        self.cohortname = cohortname
        self.female = female
        self.male = male

    # общее количество в когороте
    def total(self):
        return self.male + self.female

    # выживаемость к следующему возрастному интервалу
    def future(self, interval):
        return int(self.female * pow(self.femalemortality, interval)), int(
            self.male * pow(self.malemortality, interval))

    # предполагаемое количество детей на заданный интервал
    def babies(self, interval):
        return ((self.birthrate / self.female) * self.female) * interval

    # предполагаемое распределение мигрантов по возрастам и полу
    def migration(self, migrants, interval):
        self.male += int(((migrants * self.migrate) * interval) * 0.47)
        self.female += int(((migrants * self.migrate) * interval) * 0.53)

class DemForecasting:
    @staticmethod
    def ComponentMethod(startdata, interval, iterations):  # метод передвижки
        # инициализация популяции
        pop = []
        i = 0
        while i < len(startdata) - 3:
            pop.append(Population(startdata[i], startdata[i + 2], startdata[i + 3]))
            i += 4

        # получение коэф. смертности
        mr = pd.read_excel("morrate.xlsx")
        x = 0.04  # темп роста смертности для когорот 75 и старше
        for i in range(len(pop)):
            if i < mr.shape[0] - 1:
                pop[i].malemortality = mr.iloc[i + 1, 3]
                pop[i].femalemortality = mr.iloc[i + 1, 4]
            else:
                pop[i].malemortality = mr.iloc[mr.shape[0] - 1, 3] - x
                pop[i].femalemortality = mr.iloc[mr.shape[0] - 1, 4] - x
                x += 0.04

        # получение коэф. рождаемости
        br = pd.read_excel("birthrate.xlsx")
        m = 0
        x = 0
        while m < br.shape[0]:
            while x < len(pop):
                if br.iloc[m, 0] == pop[x].cohortname:
                    pop[x].birthrate = br.iloc[m, 1]
                    x += 1
                    m += 1
                    break
                else:
                    x += 1

        # получение коэф. миграции + миграционное сальдо

        for k in range(iterations):  # цикл прогнозных итераций
            # возрастная передвижка
            for i in reversed(range(len(pop))):
                if i == len(pop) - 1:  # последняя когорта (100 и более) умирает
                    pop[i].female = 0
                    pop[i].male = 0
                else:
                    pop[i + 1].female, pop[i + 1].male = pop[i].future(interval)

            # расчет количества новорожденных
            allbabies = 0
            for i in range(3, 10):
                allbabies += pop[i].babies(interval)

            pop[0].female = int(allbabies * 0.49)
            pop[0].male = int(allbabies * 0.51)

        # расчет общей численности
        popsize = 0
        for i in range(len(pop)):
            popsize += pop[i].total()

        return popsize

    @staticmethod
    def ExtAvgRiseAbs(olddata, interval):
        inc = 0
        for i in range(1, len(olddata)):
            inc += olddata[i] - olddata[i - 1]

        inc = inc / len(olddata)
        for i in range(interval):
            olddata.append(olddata[len(olddata) - 1] + inc)

    @staticmethod
    def ExtAvgRise(olddata, interval):
        inc = 0
        for i in range(1, len(olddata)):
            inc += (olddata[i] / olddata[i - 1]) - 1

        inc = inc / len(olddata)
        for i in range(interval):
            olddata.append(olddata[len(olddata) - 1] * (inc + 1))


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

# подготовка данных для метода передвижки (половозрастной столбец за 23 год)
fulldata23 = []
m = 3
for i in range(9, data.shape[0]):
    if data.iloc[i, data.shape[1] - 1] == data.iloc[i, data.shape[1] - 1]:
        if m == 3:
            fulldata23.append(data.iloc[i - 1, 0])
            m = 0
        fulldata23.append(data.iloc[i, data.shape[1] - 1])
        m += 1

popsize = DemForecasting.ComponentMethod(fulldata23, 5, 1)

for i in range(n):
    year.append(year[len(year) - 1] + 1)

plt.plot(year, districtdata, '.', color='black', markersize=7)
plt.plot(year[:len(districtdata) - n], districtdata[:len(districtdata) - n], color='blue', label='РОССТАТ')
plt.plot(year[len(districtdata) - n - 1:], districtdata[len(districtdata) - n - 1:], color='red',
         label='Прогноз экстрапол.')
plt.plot((2023, 2028), (districtdata[len(districtdata) - n - 1], popsize), '-ok', color='orange',
         label='Прогноз метод передвиж. (без миграции)')
plt.legend(loc='upper left')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(data.columns[0])
plt.show()
