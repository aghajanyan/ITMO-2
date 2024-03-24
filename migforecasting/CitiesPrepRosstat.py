import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode


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
        self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.funds, self.companies,
        self.factoriescap, self.conscap, self.consnewareas, self.consnewapt, self.retailturnover,
            self.foodservturnover, self.saldo)
        return result

    def __iter__(self):
        return iter([self.name, self.popsize, self.avgemployers, self.unemployed, self.avgsalary, self.livarea,
        self.beforeschool, self.docsperpop, self.bedsperpop, self.cliniccap, self.invests, self.funds, self.companies,
        self.factoriescap, self.conscap, self.consnewareas, self.consnewapt, self.retailturnover,
            self.foodservturnover, self.saldo])

data = pd.read_excel("d0.xlsx")

saldo = data.iloc[13, 1]
saldo = ''.join(saldo.split())
print(int(saldo))

examples = []
cityname = data.iloc[1, 1]
for m in range(1, data.shape[1]):
    # вычисление суммы промышленного оборота (+ корр. строки в эксель)
    factorycap = 0
    for i in range(58, 62):
        try:
            data.iloc[i, m] = ''.join(data.iloc[i, m].split())
            factorycap += float(data.iloc[i, m].replace(",", "."))
        except (ValueError, TypeError):
            factorycap += 0
        except AttributeError:
            factorycap += data.iloc[i, m]

    if data.iloc[1, m] == data.iloc[1, m]:  # если не NAN, то город меняется
        cityname = data.iloc[1, m]

    examples.append(City(cityname, data.iloc[4, m], data.iloc[15, m], data.iloc[17, m], data.iloc[19, m],
                         data.iloc[22, m], data.iloc[23, m], data.iloc[27, m], data.iloc[34, m], data.iloc[38, m],
                         data.iloc[45, m], data.iloc[52, m], data.iloc[55, m], factorycap, data.iloc[63, m],
                         data.iloc[64, m], data.iloc[65, m], data.iloc[72, m], data.iloc[74, m], data.iloc[13, m]))


