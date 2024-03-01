import pandas as pd
import numpy as np

data = pd.read_excel("data0.xlsx", 17)
cohotnames = pd.read_excel("migbyage.xlsx")

newdata = []

m = 0
k = 4
check = 0
while m < cohotnames.shape[0]:
    while k < data.shape[0]:
        if cohotnames.iloc[m, 0] == data.iloc[k, 0]:
            newdata.append(np.array(data.iloc[k]))
            newdata.append(np.array(data.iloc[k + 1]))
            newdata.append(np.array(data.iloc[k + 2]))
            newdata.append(np.array(data.iloc[k + 3]))
            m += 1
            check +=1
            break
        k += 4

print(check)
newdata = pd.DataFrame(newdata)
newdata.to_excel("newdata.xlsx")