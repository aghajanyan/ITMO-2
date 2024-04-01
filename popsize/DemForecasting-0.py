import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import seaborn


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
        return (self.birthrate * self.female) * interval

    # предполагаемое распределение мигрантов по возрастам и полу
    def migration(self, migrants):
        self.male += int((migrants * self.migrate) * 0.47)
        self.female += int((migrants * self.migrate) * 0.53)


class DemForecasting:
    @staticmethod
    def ComponentMethod(startdata, interval, iterations, regionid, regionname, inputfile, migON):  # метод передвижки
        # инициализация популяции
        pop = []
        i = 0
        while i < len(startdata) - 3:
            pop.append(Population(startdata[i], startdata[i + 2], startdata[i + 3]))
            i += 4

        # вывод стартовой половозрастной структуры (2023 год)
        h = ['Cohort', 'Female', 'Male']
        with open("" + regionname + " 2023.csv", 'w', newline='\n') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow(h)
            for b in pop:
                wr.writerow(list([b.cohortname, int(b.female), int(b.male)]))

        # получение коэф. смертности
        mr = pd.read_excel("morrate"+ inputfile +".xlsx")
        x = 0.04  # темп роста смертности для когорот 75 и старше
        for i in range(len(pop)):
            if i < mr.shape[0] - 1:
                pop[i].malemortality = mr.iloc[i + 1, 3]
                pop[i].femalemortality = mr.iloc[i + 1, 4]
            else:
                pop[i].malemortality = mr.iloc[mr.shape[0] - 1, 3] - x
                pop[i].femalemortality = mr.iloc[mr.shape[0] - 1, 4] - x
                x += 0.05

        # получение коэф. рождаемости
        br = pd.read_excel("birthrate"+ inputfile +".xlsx")
        m = 0
        x = 0
        while m < br.shape[0]:
            while x < len(pop):
                if br.iloc[m, 0] == pop[x].cohortname:
                    pop[x].birthrate = br.iloc[m, 1] / 1000
                    x += 1
                    m += 1
                    break
                else:
                    x += 1

        # получение коэф. миграции
        mr = pd.read_excel("migbyage.xlsx")
        for i in range(len(pop)):
            pop[i].migrate = mr.iloc[i, 1]

        # цикл прогнозных итераций
        popsize = []
        currentmigrants = 0
        for k in range(iterations):
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

            # расчет количества мигрантов
            if migON == True:
                datasaldo = pd.read_excel("migsaldo.xlsx")
                # вычисление среднего приращения миграционного сальдо
                migsaldo = 0.0
                for j in range(1, datasaldo.shape[1] - 1):
                    migsaldo += datasaldo.iloc[regionid, j + 1] - datasaldo.iloc[regionid, j]

                migsaldo = migsaldo / (datasaldo.shape[1] - 2)

                if k == 0:
                    currentmigrants = datasaldo.iloc[regionid, 5]

                allmig = 0
                for i in range(interval):
                    currentmigrants = currentmigrants + migsaldo
                    allmig += currentmigrants

                for i in range(len(pop)):
                    pop[i].migration(allmig)

            # расчет общей численности
            allpop = 0
            for i in range(len(pop)):
                allpop += pop[i].total()
            popsize.append(allpop)

        return popsize, pop

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

# !!ПАРАМЕТРЫ ПРОГНОЗА!!
regionid = 0 # номер региона (номер листа эксель (от 0 до 17))
iterations = 2  # количество прогнозных итераций (шаг 5 лет)

data = pd.read_excel("data0.xlsx", sheet_name=regionid)
regionname = data.columns[0]

# подготовка данных для методов экстраполяции (среднрй темп роста общей численности)
districtdata = []
year = []
for j in range(1, data.shape[1]):
    districtdata.append(data.iloc[1, j])
    year.append(int(data.iloc[0, j]))

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

popsizeRU, popsizeLO = [], []
popsizemigRU, popsizemigLO = [], []
popLO, popRU = [], []
popmigRU, popmigLO = [], []
interval = 5    # шаг прогноза (пока не трогать!!)
#popsizeRU, popRU = DemForecasting.ComponentMethod(fulldata23, interval, iterations, regionid, regionname, "RU", False)
popsizeLO, popLO = DemForecasting.ComponentMethod(fulldata23, interval, iterations, regionid, regionname, "LO", False)
popsizemigRU, popmigRU = DemForecasting.ComponentMethod(fulldata23, interval, iterations, regionid, regionname, "RU", True)
popsizemigLO, popmigLO = DemForecasting.ComponentMethod(fulldata23, interval, iterations, regionid, regionname, "LO", True)


for i in range(iterations * interval):
    year.append(year[len(year) - 1] + 1)

for i in range(len(year) - len(districtdata)):
    districtdata.append(None)

h = ['Cohort', 'Female', 'Male']
# запись данных в csv файл
with open(""+regionname+" "+ str(year[len(year) - 1]) +".csv", 'w', newline='\n') as csv_file:
    wr = csv.writer(csv_file, delimiter=',')
    wr.writerow(h)
    for a in popmigLO:
        wr.writerow(list([a.cohortname, a.female, a.male]))

# вывод полученных результатов
interval = interval * iterations
plt.plot(year, districtdata, '.', color='black', markersize=7)
plt.plot(year[:len(districtdata) - interval], districtdata[:len(districtdata) - interval], color='blue',
         label='РОССТАТ')
plt.plot(year[len(districtdata) - interval - 1:], districtdata[len(districtdata) - interval - 1:], color='red',
         label='Прогноз экстрапол.')

newdataRU, newdataLO = [], []
newdatamigRU, newdatamigLO = [], []

for i in range(len(popsizeRU) + 1):
    if i == 0:
        newdataRU.append(districtdata[len(districtdata) - interval - 1])
        newdataLO.append(districtdata[len(districtdata) - interval - 1])
        newdatamigRU.append(districtdata[len(districtdata) - interval - 1])
        newdatamigLO.append(districtdata[len(districtdata) - interval - 1])
    else:
        newdataRU.append(popsizeRU[i - 1])
        newdataLO.append(popsizeLO[i - 1])
        newdatamigRU.append(popsizemigRU[i - 1])
        newdatamigLO.append(popsizemigLO[i - 1])

newX = []
for i in range(iterations + 1):
    newX.append(year[len(year) - interval - 1])
    interval-=5

plt.plot(newX, newdataRU, '-ok', color='orange', label='Прогноз метод передвиж. РФ (без миграции)')
plt.plot(newX, newdataLO, '-ok', color='black', label='Прогноз метод передвиж. ЛО (без миграции)')
plt.plot(newX, newdatamigRU, '-ok', color='purple', label='Прогноз метод передвиж. РФ (c миграцией)')
plt.plot(newX, newdatamigLO, '-ok', color='grey', label='Прогноз метод передвиж. ЛО (c миграцией)')

plt.legend(loc='upper left')
plt.xlabel("Год")
plt.ylabel("Численность населения")
plt.title(regionname)
plt.show()

# демографические пирамиды
oldpop = []
i = 0
while i < len(fulldata23) - 3:
    oldpop.append(Population(fulldata23[i], fulldata23[i + 2], fulldata23[i + 3]))
    i += 4

cohorts = []
maleold, malenew = [], []
femaleold, femalenew = [], []
dataplotold, dataplotnew = {'Age': [], 'Male': [], 'Female': []}, {'Age': [], 'Male': [], 'Female': []}
for a in oldpop:
    cohorts.append(a.cohortname)
    femaleold.append(a.female)
    maleold.append(a.male * -1)

dataplotold['Age'] = cohorts
dataplotold['Male'] = maleold
dataplotold['Female'] = femaleold
dataplotold = pd.DataFrame(dataplotold)

for a in popmigLO:
    femalenew.append(a.female)
    malenew.append(a.male * -1)

dataplotnew['Age'] = cohorts
dataplotnew['Male'] = malenew
dataplotnew['Female'] = femalenew
dataplotnew = pd.DataFrame(dataplotnew)

ages = ['100-', '95-99', '90-94', '85-89', '80-84', '75-79', '70-74', '65-69', '60-64',
        '55-59', '50-54', '45-49', '40-44', '35-39', '30-34', '25-29', '20-24', '15-19', '10-14', '5-9', '0-4']


# две пирамиды отдельно
fig, (ax1, ax2) = plt.subplots(1, 2)

seaborn.barplot(data=dataplotold, x='Male', y='Age', order=ages, color='red', ax=ax1)
seaborn.barplot(data=dataplotold, x='Female', y='Age', order=ages, color='red', ax=ax1)

ax1.set_title(""+regionname+" 2023")
ax1.grid()
ax1.set_xlabel("Мужины | Женщины")

seaborn.barplot(data=dataplotnew, x='Male', y='Age', order=ages, color='grey', ax=ax2)
seaborn.barplot(data=dataplotnew, x='Female', y='Age', order=ages, color='grey', ax=ax2)

ax2.set_title(""+regionname+" "+ str(year[len(year) - 1]) +"")
ax2.grid()
ax2.set_xlabel("Мужины | Женщины")
plt.show()


"""
# две пирамиды вместе
ax1 = seaborn.barplot(data=dataplotold, x='Male', y='Age', order=ages, color='red')
ax2 = seaborn.barplot(data=dataplotold, x='Female', y='Age', order=ages, color='red')

plt.title(""+regionname+" 2023")
plt.grid()
plt.xlabel("Мужины | Женщины")

ax3 = seaborn.barplot(data=dataplotnew, x='Male', y='Age', order=ages, color='gray')
ax4 = seaborn.barplot(data=dataplotnew, x='Female', y='Age', order=ages, color='gray')

plt.title(""+regionname+" "+ str(year[len(year) - 1]) +"")
plt.grid()
plt.xlabel("Мужины | Женщины")
plt.show()
"""