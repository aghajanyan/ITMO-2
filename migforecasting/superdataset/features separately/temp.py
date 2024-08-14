import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("badcompanies (allmun).csv")

data = data.sort_values(by=['oktmo', 'year'])
corrset = []
i = 0
while i < data.shape[0]:
    corrset.append(data.iloc[i])
    if i != data.shape[0] - 1:
        if data.iloc[i, 0] == data.iloc[i + 1, 0] and data.iloc[i, 2] == data.iloc[i + 1, 2]:
            if data.iloc[i, 3] < data.iloc[i + 1, 3]:
                corrset.pop()
                corrset.append(data.iloc[i + 1])
            else:
                data.iloc[i + 1] = data.iloc[i]
                corrset.pop()
    i+=1

titles = ['oktmo', 'name', 'year', 'badcompanies']

corrset = pd.DataFrame(corrset, columns=titles)

corrset = corrset.drop_duplicates()

corrset.to_csv("badcompanies (allmun).csv", index=False)


print('whats up')
# удалить строку
#data = data.drop([0])
#удалить столбец
#data = data[rawdata.columns.drop('index')]

#data.to_csv("theatres (allmun).csv", index=False)