import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


male = pd.read_csv("agestruct male 2022.csv")
female = pd.read_csv("agestruct female 2022.csv")

mf = [male, female]
mf = pd.concat(mf)

mf = mf.sort_values(by=['oktmo', 'gender'])
mf = mf.reset_index(drop=True)

# каждой твари по паре, инчае дроп
a = 0
while a < len(mf) - 1:
    if mf.iloc[a]['oktmo'] != mf.iloc[a + 1]['oktmo']:
        mf = mf.drop(a)
    else:
        a += 2

# если значение 0 (замена nan'ов), приравнять кол-во предыдущего возраста
for i in range(len(mf)):
    for j in range(0, 70):
        if mf.iloc[i][''+str(j)+''] == 0:
            index = mf.columns.get_loc(''+str(j)+'')
            mf.iloc[i, index] = mf.iloc[i, index - 1]

mf.to_csv("agestruct 2022.csv", index=False)