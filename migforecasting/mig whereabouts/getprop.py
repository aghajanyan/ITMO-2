import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

saldo = pd.read_csv("saldo LO for AK.csv")

saldo['saldo'] = saldo['saldo'].abs()

saldo = np.array(saldo)

prepdata = []
for i in range(len(saldo)):
    if saldo[i, 3] == 'Межрегиональная' or saldo[i, 3] == 'Внутрирегиональная' or saldo[i, 3] == 'Международная':
        prepdata.append(saldo[i])

prepdata = np.array(prepdata)

print('ok')