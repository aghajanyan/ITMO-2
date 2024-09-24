import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

n = 2010
examples = []
for k in range(12):
    n+=1
    data = pd.read_csv("internat/saldo/saldo internat "+str(n)+" (allmun).csv")
    for i in range(data.shape[0]):
        examples.append(data.iloc[i])

examples = np.array(examples)
titles = ['oktmo', 'name', 'year', 'saldo']
examples = pd.DataFrame(examples, columns=titles)
examples.to_csv("saldo internat (allmun).csv", index=False)

"""
    #удаление NAN значений из выборки
    data = pd.read_csv("saldo/saldo "+str(n)+" (allmun).csv")
    for i in range(data.shape[0]):
        if data.iloc[i, 3] == data.iloc[i, 3]:
            examples.append(data.iloc[i])
    
    examples = np.array(examples)
    titles = ['oktmo', 'name', 'year', 'saldo']
    examples = pd.DataFrame(examples, columns=titles)
    examples.to_csv("saldo/saldo "+str(n)+" (allmun).csv", index=False)
"""

"""
    #удаление повторяющихся значений из выборки
    data = pd.read_csv("saldo/saldo " + str(n) + " (allmun).csv")
    data = data.drop_duplicates(subset='oktmo')
    data.to_csv("saldo/saldo " + str(n) + " (allmun).csv", index=False)
"""
