import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

n = 2011
for k in range(11):
    n+=1
    data = pd.read_csv("saldo/saldo "+str(n)+" (allmun).csv")
    data = data.drop_duplicates(subset='oktmo')
    data.to_csv("saldo/saldo "+str(n)+" (allmun).csv", index=False)
    