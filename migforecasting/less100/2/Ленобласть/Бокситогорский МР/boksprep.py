import csv
import numpy as np
import random
import pandas as pd


data = pd.read_excel("бокс-23.xlsx")

titles = ['popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'invests', 'factoriescap', 'conscap', 'consnewareas', 'retailturnover',
          'foodservturnover', 'saldo']
boksdata = []
boksdata = pd.DataFrame(boksdata, columns=titles)

try:
    data.iloc[128, 3] = float(data.iloc[128, 3]) / 1000
except ValueError:
    data.iloc[128, 3] = data.iloc[128, 3]

boksdata.loc[0] = [data.iloc[7, 3],
                   float(data.iloc[16, 3]) / 1000,
                   (float(data.iloc[37, 3]) / 100) * data.iloc[7, 3],
                   data.iloc[45, 3],
                   data.iloc[130, 3],
                   float(data.iloc[100, 3]) / 1000,
                   float(data.iloc[67, 3]) / 1000,
                   data.iloc[128, 3],
                   data.iloc[129, 3],
                   float(data.iloc[96, 3]) / 1000,
                   float(data.iloc[97, 3]) / 1000,
                   data.iloc[10, 3]]

print('')