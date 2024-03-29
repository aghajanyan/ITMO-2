import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import csv
import os
import docx

data = docx.Document("cities17-18/0/d1.docx")

a = data.tables[3]
b = data.tables[4]
c = data.tables[5]
d = data.tables[6]

example = []
tmp = []

for i in range(len(a.rows)):
    tmp.clear()
    for j in range(len(a.rows[i].cells)):
        tmp.append(a.rows[i].cells[j].text)

    for j in range(len(b.rows[i].cells)):
        tmp.append(b.rows[i].cells[j].text)

    example.append(np.array(tmp))

for i in range(len(c.rows)):
    tmp.clear()
    for j in range(len(c.rows[i].cells)):
        tmp.append(c.rows[i].cells[j].text)

    for j in range(len(d.rows[i].cells)):
        tmp.append(d.rows[i].cells[j].text)

    example.append(np.array(tmp))

example = np.array(example)

result = pd.DataFrame(example)

result.to_excel("007d.xlsx")