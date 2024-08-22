import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

saldo = pd.read_csv("saldo LO for AK.csv")

saldo['saldo'] = saldo['saldo'].abs()
saldo = saldo.dropna()

saldo = np.array(saldo)

inregmas = []
outregmas = []
outmas = []
for i in range(len(saldo)):
    if saldo[i, 3] == 'Межрегиональная':
        outregmas.append(saldo[i])
    if saldo[i, 3] == 'Внутрирегиональная':
        inregmas.append(saldo[i])
    if saldo[i, 3] == 'Международная':
        outmas.append(saldo[i])

titles = ['oktmo', 'name', 'year', 'whereabouts', 'saldo']
outregmas = pd.DataFrame(outregmas, columns=titles)
inregmas = pd.DataFrame(inregmas, columns=titles)
outmas = pd.DataFrame(outmas, columns=titles)

outregmas = outregmas.sort_values(by=['oktmo', 'year'])


inregavg = 0
outregavg = 0
outavg = 0

total = outregmas['saldo'].sum() + inregmas['saldo'].sum() + outmas['saldo'].sum()
inregavg = inregmas['saldo'].sum() / total
outregavg = outregmas['saldo'].sum() / total
outavg = outmas['saldo'].sum() / total


print('ok')

features = ['oktmo', 'name', 'regional', 'national', 'international']