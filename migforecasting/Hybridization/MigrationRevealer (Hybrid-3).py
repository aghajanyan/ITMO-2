import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#перевод из нормализованного диапазона в реальное
maxsaldoP = 10001    # dataset 20 (also positive flow)
maxsaldoN = 2854     # negative flow (dataset 20)

positive = pd.read_csv("superdataset-21 (positive flow).csv")
negative = pd.read_csv("superdataset-21 (negative flow).csv")

negative = negative[negative.columns.drop('consnewareas')]
positive = positive[positive.columns.drop('consnewareas')]

# перетасовка
negative = negative.sample(frac=1)
positive = positive.sample(frac=1)

#одинаковая нормализация сальдо
#for i in range(len(negative)):
#    negative.iloc[i, 0] = negative.iloc[i, 0] * maxsaldoN
#    negative.iloc[i, 0] = negative.iloc[i, 0] / maxsaldoP

# разбиение датасета на входные признаки и выходной результат (сальдо)
# 1 - для оттока (негатив), 2 - для притока (позитив), 3 - для классификатора, 4 - для гибридной модели (только тест)
datasetin1 = np.array(negative[negative.columns.drop('saldo')])
datasetin2 = np.array(positive[positive.columns.drop('saldo')])
datasetout1 = np.array(negative[['saldo']])
datasetout2 = np.array(positive[['saldo']])

# разбиение на обучающую и тестовую выборку
trainin1, testin1, trainout1, testout1 = train_test_split(datasetin1, datasetout1, test_size=0.2, random_state=146)
trainin2, testin2, trainout2, testout2 = train_test_split(datasetin2, datasetout2, test_size=0.2, random_state=146)

#подготовка входных и выходных данных для обучения классификатора
#объединить входы негатива и позитива
classdata = []
for i in range(trainin1.shape[0]):
    classdata.append(np.append(trainin1[i], 0))  # 0 - для отрицательного сальдо
    if i < trainin2.shape[0] - 1:
        classdata.append(np.append(trainin2[i], 1))  # 1 - для подожительного сальдо

classdata = pd.DataFrame(classdata)
classdata = classdata.sample(frac=1)    # перетасовка

# разделение входных и выходных данных
trainout3 = classdata[[classdata.shape[1] - 1]]
classdata = classdata[classdata.columns.drop(classdata.shape[1] - 1)]
trainin3 = classdata

trainin3 = np.array(trainin3)
trainout3 = np.array(trainout3)

#тоже самое для тестовой выборки + формирование теста для гибридной модели (4)
classdatatest = []
for i in range(testin1.shape[0]):
    classdatatest.append(np.append(testin1[i], [-abs(testout1[i, 0]), 0]))
    if i < testin2.shape[0] - 1:
        classdatatest.append(np.append(testin2[i], [testout2[i, 0], 1]))

classdatatest = pd.DataFrame(classdatatest)
classdatatest = classdatatest.sample(frac=1)

testout3 = classdatatest[[classdatatest.shape[1] - 1]]
testout4 = classdatatest[[classdatatest.shape[1] - 2]]
classdatatest = classdatatest[classdatatest.columns.drop(classdatatest.shape[1] - 1)]
classdatatest = classdatatest[classdatatest.columns.drop(classdatatest.shape[1] - 1)]
testin3 = classdatatest

testin3 = np.array(testin3)
testout3 = np.array(testout3)   # для проверки классификатора
testout4 = np.array(testout4)   # для проверки гибридной модели

# модель 1 (прогноз отрицательного сальдо)
model1 = RandomForestRegressor(n_estimators=100, random_state=0)
model1.fit(trainin1, trainout1.ravel())

# модель 2 (прогноз положительного сальдо)
model2 = RandomForestRegressor(n_estimators=100, random_state=0)
model2.fit(trainin2, trainout2.ravel())

# модель 3 (классифкация отрицательного/положительного сальдо)
model3 = RandomForestClassifier(n_estimators=100, random_state=0)
model3.fit(trainin3, trainout3.ravel())
#predtrainclass = model3.predict(trainin3)

# оценка тестовой выборки
# прогноз регресионных моделей на общей тестовой выборке (для классификатора)
prednegative = model1.predict(testin3)
predpositive = model2.predict(testin3)

predclass = model3.predict(testin3)
classerror = f1_score(testout3, predclass)
print("f1 score: ", classerror)

fpr, tpr, thresholds = roc_curve(testout3, predclass)
aucscore = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % aucscore)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('Прогноз оттока или притока (тестовый сет)')
plt.legend(loc="lower right")
plt.show()

#получение ответа гибридной модели + преобразование в естественный вид
hybridpred = []
for i in range(len(prednegative)):
    if int(predclass[i]) == 1:
        hybridpred.append(predpositive[i] * maxsaldoP)
    else:
        hybridpred.append((prednegative[i] * maxsaldoN) * -1)

hybridpred = np.array(hybridpred)

#преобразовать тестовый датасет и потом сравнить с ответом гибридной модели
for i in range(len(testout4)):
    if testout4[i] > 0:
        testout4[i] = testout4[i] * maxsaldoP
    else:
        testout4[i] = testout4[i] * maxsaldoN

hybriderror = mean_absolute_error(testout4, hybridpred)

print('Error of hybrid model: ', hybriderror)



