import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

alltogether = []

for y in range(2011, 2024):
    data = pd.read_csv("agestruct (prop)/agestruct "+ str(y) +" (prop).csv")
    alltogether += [data]

alltogether = pd.concat(alltogether)
alltogether = alltogether.sort_values(by=['oktmo', 'year'])
alltogether = alltogether.reset_index(drop=True)

alltogether.to_csv("agestruct prop.csv", index=False)

print('Done')