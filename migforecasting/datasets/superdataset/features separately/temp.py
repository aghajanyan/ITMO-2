import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("theatres (allmun).csv")

# удалить строку
data = data.drop([0])
#удалить столбец
#data = data[rawdata.columns.drop('index')]

data.to_csv("theatres (allmun).csv", index=False)