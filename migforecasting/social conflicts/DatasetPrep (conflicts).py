import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


inputdata = pd.read_csv('superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict, no output).csv')
output = pd.read_excel('Conflict assessment.xlsx')

output = output.sort_values(by=['oktmo', 'year'])

output['sum'] = output['sum'] / 12

inputdata['risk'] = list(output['sum'])

examples = []
for i in range(len(inputdata) - 1):
    if inputdata.iloc[i, 0] == inputdata.iloc[i + 1, 0]:
        if inputdata.iloc[i, 1] + 1 == inputdata.iloc[i + 1, 1]:
            inputdata.iloc[i, inputdata.shape[1] - 1] = inputdata.iloc[i + 1, inputdata.shape[1] - 1]
            examples.append(inputdata.iloc[i])


examples = np.array(examples)
features = inputdata.columns
examples = pd.DataFrame(examples, columns=features)

examples = examples.drop(columns=['oktmo', 'name', 'year'])
examples.to_csv('ready-superdataset-24-alltime-clust (IQR)-normbysoul-f (conflict).csv', index=False)
