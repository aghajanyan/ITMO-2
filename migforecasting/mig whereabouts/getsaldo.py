import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

inflow = pd.read_csv("inflow LO.csv")
outflow = pd.read_csv("outflow LO.csv")

outflow = outflow[outflow.columns.drop('name')]

inflow = inflow.merge(outflow, on=['oktmo', 'year', 'whereabouts'], how='left')

inflow = np.array(inflow)

saldo = []
tmp = []
for i in range(len(inflow)):
    tmp.append(inflow[i, 0])
    tmp.append(inflow[i, 1])
    tmp.append(inflow[i, 2])
    tmp.append(inflow[i, 3] - inflow[i, 4])
    saldo.append(np.array(tmp))
    tmp.clear()

titles = ['oktmo', 'name', 'year', 'saldo']

saldo = pd.DataFrame(saldo, columns=titles)

saldo.to_csv("saldo 2008 (allmun).csv", index=False)

print('ok')