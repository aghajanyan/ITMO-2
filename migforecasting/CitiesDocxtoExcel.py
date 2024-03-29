import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import csv
import os
import docx

data = docx.Document("0.docx")

example = []
tmp = []

if len(data.tables) == 7:    # данные разбиты на 4 таблицы
    a = data.tables[3]
    b = data.tables[4]
    c = data.tables[5]
    d = data.tables[6]

    for i in range(len(a.rows)):
        tmp.clear()
        for j in range(len(a.rows[i].cells)):
            tmp.append(a.rows[i].cells[j].text)

        for j in range(len(b.rows[i].cells)):
            tmp.append(b.rows[i].cells[j].text)

        example.append(np.array(tmp))

    for i in range(2, len(c.rows)):
        tmp.clear()
        for j in range(len(c.rows[i].cells)):
            tmp.append(c.rows[i].cells[j].text)

        for j in range(len(d.rows[i].cells)):
            tmp.append(d.rows[i].cells[j].text)

        example.append(np.array(tmp))
elif len(data.tables) == 5:   # данные разбиты на 2 таблицы (моногород)
    a = data.tables[3]
    b = data.tables[4]
    for i in range(len(a.rows)):
        tmp.clear()
        for j in range(len(a.rows[i].cells)):
            tmp.append(a.rows[i].cells[j].text)

        example.append(np.array(tmp))

    for i in range(1, len(b.rows)):
        tmp.clear()
        if len(b.rows[i].cells) == 4:
            for j in range(len(b.rows[i].cells)):
                tmp.append(b.rows[i].cells[j].text)
        else:
            tmp.append(b.rows[i].cells[0].text)
            tmp.append(b.rows[i].cells[2].text)
            tmp.append(b.rows[i].cells[5].text)
            tmp.append(b.rows[i].cells[6].text)

        example.append(np.array(tmp))

elif len(data.tables) == 6:
    v = 2
    for k in range(3, 6):
        if k == 3:
            v = 0
        else:
            v = 2
        for i in range(v, len(data.tables[k].rows)):
            tmp.clear()
            for j in range(len(data.tables[k].rows[i].cells)):
                tmp.append(data.tables[k].rows[i].cells[j].text)

            example.append(np.array(tmp))


example = np.array(example)

result = pd.DataFrame(example)

result.to_excel("d0.xlsx")