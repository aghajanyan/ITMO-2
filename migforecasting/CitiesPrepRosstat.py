import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import csv


class City:
    def __init__(self, name, popsize, avgemployers, unemployed, avgsalary, livarea, beforeschool, docsperpop,
                 bedsperpop, cliniccap, invests, funds, companies, factoriescap, conscap, consnewareas, consnewapt,
                 retailturnover, foodservturnover, saldo):
        self.name = name
        self.popsize = popsize
        self.avgemployers = avgemployers
        self.unemployed = unemployed
        self.avgsalary = avgsalary
        self.livarea = livarea
        self.beforeschool = beforeschool
        self.docsperpop = docsperpop
        self.bedsperpop = bedsperpop
        self.cliniccap = cliniccap
        self.invests = invests
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
        result = "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}".format(
            self.name, self.popsize, self.avgemployers, self.unemployed, self.avgsalary, self.livarea,
            self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.funds,
            self.companies,
            self.factoriescap, self.conscap, self.consnewareas, self.consnewapt, self.retailturnover,
            self.foodservturnover, self.saldo)
        return result

    def __iter__(self):
        return iter([self.name, self.popsize, self.avgemployers, self.unemployed, self.avgsalary, self.livarea,
                     self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.funds,
                     self.companies,
                     self.factoriescap, self.conscap, self.consnewareas, self.consnewapt, self.retailturnover,
                     self.foodservturnover, self.saldo])

examples = []
for dis in range(19):
    data = pd.read_excel("cities19-21/d2.xlsx")

    x = 0  # для моногородних файлов
    if data.shape[1] == 4:
        x = -2  # для одногородних файлов

    # нормализация (убрать неразрывный пробел и запятые вещественных чисел)
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            try:
                data.iloc[i, j] = ''.join(data.iloc[i, j].split())
                data.iloc[i, j] = data.iloc[i, j].replace(",", ".")
            except AttributeError:
                data.iloc[i, j] = data.iloc[i, j]

    if x == -2:
        cityname = data.iloc[0, 0]
    else:
        cityname = data.iloc[1, 1]

    for m in range(1, data.shape[1]):
        # вычисление суммы промышленного оборота (+ корр. строки в эксель)
        factorycap = 0
        for i in range(58 + x, 62 + x):
            try:
                factorycap += float(data.iloc[i, m])
            except (ValueError, TypeError):
                factorycap += 0

        if data.iloc[1, m] == data.iloc[1, m]:  # если не NAN, то город меняется
            cityname = data.iloc[1, m]

        examples.append(City(cityname, data.iloc[4 + x, m], data.iloc[15 + x, m], data.iloc[17 + x, m],
                             data.iloc[19 + x, m], data.iloc[22 + x, m], data.iloc[23 + x, m], data.iloc[27 + x, m],
                             data.iloc[34 + x, m], data.iloc[38 + x, m], data.iloc[45 + x, m], data.iloc[52 + x, m],
                             data.iloc[55 + x, m], factorycap, data.iloc[63 + x, m], data.iloc[64 + x, m],
                             data.iloc[65 + x, m], data.iloc[72 + x, m], data.iloc[74 + x, m], data.iloc[13 + x, m]))

# запись в csv
titles = ['name', 'popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea', 'beforeschool', 'docsperpop',
          'bedsperpop', 'cliniccap', 'invests', 'funds', 'companies', 'factoriescap', 'conscap', 'consnewareas',
          'consnewapt', 'retailturnover', 'foodservturnover', 'saldo']

with open("1.csv", 'w', newline='\n') as csv_file:
    wr = csv.writer(csv_file, delimiter=',')
    wr.writerow(titles)
    for a in examples:
        wr.writerow(list(a))