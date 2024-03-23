import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode

data = pd.read_excel("d0.xlsx")

saldo = data.iloc[13, 1]
saldo = ''.join(saldo.split())
print(int(saldo))
