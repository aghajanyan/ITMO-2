import csv
import numpy as np
import random
import pandas as pd


data = pd.read_excel("бокс-16-1.xlsx")

titles = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'invests', 'factoriescap', 'conscap', 'consnewareas', 'retailturnover',
          'foodservturnover', 'saldo']
boksdata = []
boksdata = pd.DataFrame(boksdata, columns=titles)

try:
    data.iloc[128, 3] = float(data.iloc[128, 3]) / 1000
except ValueError:
    data.iloc[128, 3] = data.iloc[128, 3]

x = 1
boksdata.loc[0] = [float(data.iloc[7 + x, 3]) / 1000,
                   float(data.iloc[16 + x, 3]) / 1000,
                   (float(data.iloc[31, 3]) / 100) * data.iloc[7 + x, 3],
                   data.iloc[39, 3],
                   data.iloc[105, 3],
                   float(data.iloc[83, 3]) / 1000,
                   float(data.iloc[54, 3]) / 1000,
                   data.iloc[103, 3], #float(data.iloc[103, 3]) / 1000,
                   data.iloc[104, 3],
                   float(data.iloc[79, 3]) / 1000,
                   data.iloc[80, 3], #float(data.iloc[80, 3]) / 1000,
                   data.iloc[10 + x, 3]]

boksdata.to_excel("1.xlsx")

print('')