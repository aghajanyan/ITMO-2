import csv
import numpy as np
import random
import pandas as pd

rawdata = pd.read_excel("aptprices-raw.xlsx")

avg, count = 0, 0
data, tmp = [], []
for i in range(3, rawdata.shape[0]):
    avg = 0
    count = 0
    tmp.append(rawdata.iloc[i, 1])
    for j in range(3, rawdata.shape[1]):
        if count != 4:
            try:
                avg += rawdata.iloc[i, j]
                count += 1
            except ValueError:
                avg += 0
                count += 1
        if count == 4:
            avg = avg / 4
            tmp.append(avg)
            avg = 0
            count = 0
    data.append(np.array(tmp))
    tmp.clear()

data = np.array(data)

data = pd.DataFrame(data)

data.to_excel("AvgAptPrices.xlsx")

print('done')