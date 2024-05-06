import csv
import numpy as np
import random
import pandas as pd

data = pd.read_excel("citiesless100 (raw data).xlsx")

titles = ['name', 'year', 'popsize', 'avgemployers', 'unemployed', 'avgsalary', 'livarea',
          'beforeschool', 'docsperpop', 'bedsperpop', 'cliniccap',
          'invests', 'funds', 'companies', 'factoriescap',
          'conscap', 'consnewareas', 'consnewapt', 'retailturnover',
          'foodservturnover', 'saldo']

data = pd.DataFrame(data, columns=titles)
data.to_csv("BoksArea.csv", index=False)