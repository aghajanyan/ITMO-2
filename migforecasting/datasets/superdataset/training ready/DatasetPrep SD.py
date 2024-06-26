import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def normbymax(trainset):
    tmpp = []
    for k in range(len(trainset[0])):
        maxi = trainset[0][k]
        for i in range(len(trainset)):
            if (maxi < trainset[i][k]):
                maxi = trainset[i][k]

        tmpp.append(maxi)

        for j in range(len(trainset)):
            trainset[j][k] = trainset[j][k] / maxi
    return trainset

def normbydollar(trainset, rubfeatures):
    # разделить рублевые признаки на стоимость доллара
    dollar = pd.read_csv("dollaravg.csv")
    trainset = trainset.merge(dollar, on='year', how='left')
    d = trainset[['dollar']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) / d.iloc[i, 0]
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    return trainset


def normbyinf(trainset, rubfeatures):
    # умножить рублевые признаки на соответствующую долю инфляции
    inflation = pd.read_csv("inflation1.csv")
    trainset = trainset.merge(inflation, on='year', how='left')
    inf = trainset[['inf']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                infnorm = 1 - (inf.iloc[i, 0] / 100)
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) * infnorm
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    trainset = trainset[trainset.columns.drop('inf')]
    return trainset


def normbyoil(trainset, rubfeatures):
    # умножить рублевые признаки на цену за нефть как долю процента
    oil2 = pd.read_csv("oilpricesavg.csv")
    trainset = trainset.merge(oil2, on='year', how='left')
    o = trainset[['oil']]
    for k in range(len(rubfeatures)):
        tmp = trainset[[rubfeatures[k]]]
        for i in range(len(tmp)):
            try:
                oilnorm = o.iloc[i, 0] / 100
                tmp.iloc[i, 0] = float(tmp.iloc[i, 0]) * oilnorm
            except ValueError:
                tmp.iloc[i, 0] = tmp.iloc[i, 0]
        trainset[rubfeatures[k]] = tmp
        tmp = pd.DataFrame(None)
    trainset = trainset[trainset.columns.drop('oil')]
    return trainset


rubfeatures = ['avgsalary', 'invests', 'factoriescap', 'conscap', 'retailturnover', 'foodservturnover']

