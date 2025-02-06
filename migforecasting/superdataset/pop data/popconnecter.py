import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


male = pd.read_csv("agestruct male 2023.csv")
female = pd.read_csv("agestruct female 2023.csv")

mf = [male, female]
mf = pd.concat(mf)

mf = mf.sort_values(by=['oktmo', 'gender'])
mf = mf.reset_index(drop=True)

a = 0
while a < len(mf) - 1:
    if mf.iloc[a]['oktmo'] != mf.iloc[a + 1]['oktmo']:
        mf = mf.drop(a)
    else:
        a += 2

mf.to_csv("agestruct 2023.csv", index=False)