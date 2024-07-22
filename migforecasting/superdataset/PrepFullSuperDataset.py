import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

features = ['saldo', 'cultureorg', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover',
            'foodservturnover', 'consnewareas', 'consnewapt', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
            'livestock', 'harvest', 'agrprod', 'invest', 'budincome', 'funds', 'museums', 'parks',
            'theatres', 'hospitals', 'cliniccap', 'beforeschool', 'schoolnum', 'naturesecure', 'factoriescap']

# получение сальдо. будущий выходной результат
dataset = pd.read_csv('features separately/'+features[0]+' (allmun).csv')

#соединение всех признаков к миграционному сальдо
tempset = []
for k in range(len(features) - 1):
    tempset = pd.read_csv('features separately/'+features[k + 1]+' (allmun).csv')
    tempset = tempset[tempset.columns.drop('name')]
    dataset = dataset.merge(tempset, on=['oktmo', 'year'], how='left')

dataset = dataset.drop_duplicates()

dataset.to_csv("superdataset (full data).csv", index=False)

print('Done')