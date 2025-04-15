import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


subclusters = pd.read_excel("medians of subclusters.xlsx")
data = pd.read_excel("input to change.xlsx")

for i in range(len(data)):
    for j in range((len(subclusters))):
        if data.iloc[i]['clust'] == subclusters.iloc[j]['clust']:
            for k in range(data.shape[1]):
                if data.iloc[i][k] < subclusters.iloc[j][k]:
                    data.iloc[i][k] = subclusters.iloc[j][k]


data.to_excel("input to change.xlsx")