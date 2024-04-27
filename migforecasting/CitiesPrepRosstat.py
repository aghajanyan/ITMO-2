import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import codecs
import csv
import os


class City:
    def __init__(self, name, year, popsize, avgemployers, unemployed, avgsalary, livarea, beforeschool, docsperpop,
                 bedsperpop, cliniccap, invests, orgfunds, funds, companies, factoriescap, conscap, consnewareas,
                 consnewapt, retailturnover, foodservturnover, saldo):
        self.name = name
        self.year = year
        self.popsize = popsize
        self.avgemployers = avgemployers
        self.unemployed = unemployed
        self.avgsalary = avgsalary
        self.livarea = livarea
        self.beforeschool = beforeschool
        self.docsperpop = docsperpop
        self.bedsperpop = bedsperpop
        self.cliniccap = cliniccap
        try:
            self.invests = float(invests) / float(popsize)  # доля инвест на чел.
        except ValueError:
            self.invests = invests
        self.orgfunds = orgfunds
        self.funds = funds
        self.companies = companies
        self.factoriescap = factoriescap
        self.conscap = conscap
        self.consnewareas = consnewareas
        self.consnewapt = consnewapt
        self.retailturnover = retailturnover
        self.foodservturnover = foodservturnover
        self.saldo = saldo

    def __str__(self):
        result = ("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, "
                  "{12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}").format(
            self.name, self.year, self.popsize, self.avgemployers, self.unemployed, self.avgsalary, self.livarea,
            self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.orgfunds, self.funds,
            self.companies, self.factoriescap, self.conscap, self.consnewareas, self.consnewapt, self.retailturnover,
            self.foodservturnover, self.saldo)
        return result

    def __iter__(self):
        return iter(
            [self.name, self.year, self.popsize, self.avgemployers, self.unemployed, self.avgsalary, self.livarea,
             self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.orgfunds, self.funds,
             self.companies, self.factoriescap, self.conscap, self.consnewareas, self.consnewapt,
             self.retailturnover, self.foodservturnover, self.saldo])


examples = []

for dis in range(8):
    files = next(os.walk("cities19-21/" + str(dis) + ""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities19-21/" + str(dis) + "/d" + str(f + 1) + ".xlsx")

        x = 0  # несколько городов в файле
        if data.shape[1] == 4:
            x = -2  # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(2, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -2:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[1, 1]

        for m in range(1, data.shape[1]):
            # вычисление суммы промышленного оборота (+ корр. строки в эксель)
            factorycap = 0
            for i in range(58 + x, 62 + x):
                try:
                    if data.iloc[i, m] == data.iloc[i, m]:
                        factorycap += float(data.iloc[i, m])
                except (ValueError, TypeError):
                    factorycap += 0

            if data.iloc[1, m] == data.iloc[1, m]:  # если не NAN, то город меняется
                cityname = data.iloc[1, m]

            examples.append(City(cityname, data.iloc[2 + x, m], data.iloc[4 + x, m], data.iloc[15 + x, m],
                                 data.iloc[17 + x, m], data.iloc[19 + x, m], data.iloc[22 + x, m], data.iloc[23 + x, m],
                                 data.iloc[27 + x, m], data.iloc[34 + x, m], data.iloc[38 + x, m], data.iloc[44 + x, m],
                                 data.iloc[50 + x, m], data.iloc[52 + x, m], data.iloc[55 + x, m], factorycap, data.iloc[63 + x, m],
                                 data.iloc[64 + x, m], data.iloc[65 + x, m], data.iloc[72 + x, m], data.iloc[74 + x, m],
                                 data.iloc[13 + x, m]))

for dis in range(8):
    files = next(os.walk("cities17-18/"+ str(dis) +""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities17-18/"+ str(dis) +"/d"+ str(f + 1) +".xlsx")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[data.shape[1] - 1], axis=1)

        x = -1 # несколько городов в файле
        if data.shape[1] == 3:
            x = -2  # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -2:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[0, 1]

        for m in range(1, data.shape[1]):
            if str(data.iloc[1, m]) != '2019':
                # вычисление суммы промышленного оборота (+ корр. строки в эксель)
                factorycap = 0
                for i in range(58 + x, 62 + x):
                    try:
                        if data.iloc[i, m] == data.iloc[i, m]:
                            factorycap += float(data.iloc[i, m])
                    except (ValueError, TypeError):
                        factorycap += 0

                if data.iloc[0, m] != cityname:
                    cityname = data.iloc[0, m]

                #17-18
                examples.append(City(cityname, data.iloc[1, m], data.iloc[4 + x, m], data.iloc[15 + x, m], data.iloc[17 + x, m],
                                     data.iloc[19 + x, m], data.iloc[22 + x, m], data.iloc[23 + x, m], data.iloc[27 + x, m],
                                     data.iloc[34 + x, m], data.iloc[38 + x, m], data.iloc[44 + x, m],
                                     data.iloc[50 + x, m], data.iloc[52 + x, m],
                                     data.iloc[55 + x, m], factorycap, data.iloc[63 + x, m], data.iloc[64 + x, m],
                                     data.iloc[65 + x, m], data.iloc[72 + x, m], data.iloc[74 + x, m],
                                     data.iloc[13 + x, m]))

for dis in range(8):
    files = next(os.walk("cities15-16/"+ str(dis) +""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities15-16/"+ str(dis) +"/d"+ str(f + 1) +".xlsx")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[data.shape[1] - 1], axis=1)

        x = 0 # несколько городов в файле
        if data.shape[1] == 3:
            x = -1 # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -1:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[0, 1]

        for m in range(1, data.shape[1]):
            if str(data.iloc[1, m]) != '2017':
                # вычисление суммы промышленного оборота (+ корр. строки в эксель)
                factorycap = 0
                for i in range(61 + x, 65 + x):
                    try:
                        if data.iloc[i, m] == data.iloc[i, m]:
                            factorycap += float(data.iloc[i, m])
                    except (ValueError, TypeError):
                        factorycap += 0

                if data.iloc[0, m] != cityname:
                    cityname = data.iloc[0, m]

                examples.append(City(cityname, data.iloc[1, m], data.iloc[3 + x, m], data.iloc[14 + x, m], data.iloc[16 + x, m],
                                     data.iloc[18 + x, m], data.iloc[21 + x, m], data.iloc[25 + x, m], data.iloc[28 + x, m],
                                     data.iloc[35 + x, m], data.iloc[39 + x, m], data.iloc[47 + x, m],
                                     data.iloc[53 + x, m], data.iloc[55 + x, m],
                                     data.iloc[58 + x, m], factorycap, data.iloc[66 + x, m], data.iloc[67 + x, m],
                                     data.iloc[68 + x, m], data.iloc[75 + x, m], data.iloc[77 + x, m],
                                     data.iloc[12 + x, m]))

for dis in range(8):
    files = next(os.walk("cities13-14/"+ str(dis) +""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities13-14/"+ str(dis) +"/d"+ str(f + 1) +".xlsx")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[data.shape[1] - 1], axis=1)

        x = 0 # несколько городов в файле
        if data.shape[1] == 3:
            x = -1 # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -1:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[0, 1]

        for m in range(1, data.shape[1]):
            if str(data.iloc[1, m]) != '2015':
                # вычисление суммы промышленного оборота (+ корр. строки в эксель)
                factorycap = 0
                for i in range(55 + x, 58 + x):
                    try:
                        if data.iloc[i, m] == data.iloc[i, m]:
                            factorycap += float(data.iloc[i, m])
                    except (ValueError, TypeError):
                        factorycap += 0

                if data.iloc[0, m] != cityname:
                    cityname = data.iloc[0, m]

                examples.append(City(cityname, data.iloc[1, m], data.iloc[3 + x, m], data.iloc[14 + x, m], data.iloc[16 + x, m],
                                     data.iloc[18 + x, m], data.iloc[21 + x, m], data.iloc[25 + x, m], data.iloc[28 + x, m],
                                     data.iloc[35 + x, m], data.iloc[39 + x, m], data.iloc[73 + x, m],
                                     data.iloc[43 + x, m], data.iloc[45 + x, m],
                                     data.iloc[48 + x, m], factorycap, data.iloc[59 + x, m], data.iloc[60 + x, m],
                                     data.iloc[61 + x, m], data.iloc[68 + x, m], data.iloc[70 + x, m],
                                     data.iloc[12 + x, m]))

for dis in range(8):
    files = next(os.walk("cities11-12/"+ str(dis) +""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities11-12/"+ str(dis) +"/d"+ str(f + 1) +".xlsx")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[data.shape[1] - 1], axis=1)

        x = 0 # несколько городов в файле
        if data.shape[1] == 3:
            x = -1 # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -1:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[0, 1]

        for m in range(1, data.shape[1]):
            if str(data.iloc[1, m]) != '2013':
                # вычисление суммы промышленного оборота (+ корр. строки в эксель)
                factorycap = 0
                for i in range(55 + x, 58 + x):
                    try:
                        if data.iloc[i, m] == data.iloc[i, m]:
                            factorycap += float(data.iloc[i, m])
                    except (ValueError, TypeError):
                        factorycap += 0

                if data.iloc[0, m] != cityname:
                    cityname = data.iloc[0, m]

                examples.append(City(cityname, data.iloc[1, m], data.iloc[3 + x, m], data.iloc[14 + x, m], data.iloc[16 + x, m],
                                     data.iloc[18 + x, m], data.iloc[21 + x, m], data.iloc[25 + x, m], data.iloc[28 + x, m],
                                     data.iloc[35 + x, m], data.iloc[39 + x, m], data.iloc[73 + x, m],
                                     data.iloc[43 + x, m], data.iloc[45 + x, m],
                                     data.iloc[48 + x, m], factorycap, data.iloc[59 + x, m], data.iloc[61 + x, m],
                                     data.iloc[62 + x, m], data.iloc[68 + x, m], data.iloc[70 + x, m],
                                     data.iloc[12 + x, m]))

for dis in range(8):
    files = next(os.walk("cities10/"+ str(dis) +""))
    for f in range(len(files[2])):
        data = pd.read_excel("cities10/"+ str(dis) +"/d"+ str(f + 1) +".xlsx")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[data.shape[1] - 1], axis=1)

        x = 0 # несколько городов в файле
        if data.shape[1] == 2:
            x = -1 # для моногородних файлов

        # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                try:
                    data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
                except AttributeError:
                    data.iloc[i, j] = data.iloc[i, j]

        # убираем сноски из данных
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if ')' in str(data.iloc[i, j]):
                    tmp = ''.join(data.iloc[i, j].split())
                    data.iloc[i, j] = tmp[:-2]

        if x == -1:
            cityname = data.iloc[0, 0]
        else:
            cityname = data.iloc[0, 1]

        for m in range(1, data.shape[1]):
            if str(data.iloc[1, m]) != '2011':
                # вычисление суммы промышленного оборота (+ корр. строки в эксель)
                factorycap = 0
                for i in range(55 + x, 58 + x):
                    try:
                        if data.iloc[i, m] == data.iloc[i, m]:
                            factorycap += float(data.iloc[i, m])
                    except (ValueError, TypeError):
                        factorycap += 0

                if data.iloc[0, m] != cityname:
                    cityname = data.iloc[0, m]

                examples.append(City(cityname, data.iloc[1, m], data.iloc[3 + x, m], data.iloc[14 + x, m], data.iloc[16 + x, m],
                                     data.iloc[18 + x, m], data.iloc[21 + x, m], data.iloc[25 + x, m], data.iloc[28 + x, m],
                                     data.iloc[35 + x, m], data.iloc[39 + x, m], data.iloc[73 + x, m],
                                     data.iloc[43 + x, m], data.iloc[45 + x, m],
                                     data.iloc[48 + x, m], factorycap, data.iloc[59 + x, m], data.iloc[61 + x, m],
                                     data.iloc[62 + x, m], data.iloc[68 + x, m], data.iloc[70 + x, m],
                                     data.iloc[12 + x, m]))

# запись в csv
titles = ['name', 'year', 'popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
          'invests', 'orgfunds', 'funds', 'companies', 'factoriescap',
          'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
          'foodservturnover', 'saldo']

examples = pd.DataFrame(examples, columns=titles)
examples.to_csv("citiesdataset 10-21 (+f).csv", index=False)
