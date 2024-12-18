import csv
import numpy as np
import pandas as pd

rawdata = pd.read_csv("citiesdataset-NYDcor-4 (KZ).csv")

for i in range(rawdata.shape[0]):
    if rawdata.iloc[i, rawdata.shape[1] - 1] > 0:
        rawdata.iloc[i, rawdata.shape[1] - 1] = int(1)
    else:
        rawdata.iloc[i, rawdata.shape[1] - 1] = int(0)

rawdata = rawdata.iloc[:3000]

rawdata.to_csv("citiesdataset-NYDcor-4 (CLASS-KZ-3000).csv", index=False)