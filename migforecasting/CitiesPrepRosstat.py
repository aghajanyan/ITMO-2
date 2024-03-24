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

tmp = []
examples = []

for k in range(7):
    for j in range(1, data.shape[1]):
        tmp.append(data.iloc[1, 1])
        tmp.append(data.)





