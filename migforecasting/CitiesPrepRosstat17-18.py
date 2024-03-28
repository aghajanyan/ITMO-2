import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import csv
import os
import docx

data = docx.Document("cities17-18/0/d1.docx")

a = data.tables[6]

example = []
tmp = []
for row in a.rows:
    tmp.clear()
    for cell in row.cells:
        tmp.append(cell.text)
    example.append(np.array(tmp))



example = np.array(example)

result = pd.DataFrame(example)

result.to_excel("12d.xlsx")