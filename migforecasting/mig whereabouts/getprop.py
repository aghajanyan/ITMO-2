import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

saldo = pd.read_csv("saldo LO for AK.csv")

saldo['saldo'] = saldo['saldo'].abs()
saldo = saldo.dropna()

saldo = np.array(saldo)

prepdata = []
total = 0
inreg = 0
outreg = 0
out = 0
for i in range(len(saldo)):
    if saldo[i, 3] == 'Межрегиональная':
        outreg += saldo[i, 4]
        total += saldo[i, 4]
    if saldo[i, 3] == 'Внутрирегиональная':
        inreg += saldo[i, 4]
        total += saldo[i, 4]
    if saldo[i, 3] == 'Международная':
        out += saldo[i, 4]
        total += saldo[i, 4]

inregavg = inreg / total
outreg = outreg / total
out = out / total

print('ok')