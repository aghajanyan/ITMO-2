import csv
import math

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from distributed.utils import palette

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import colormaps as cmaps
#import shap


# вывод графика с поселениями на карте согласно их реальным координатам
def townsmap():
    best = pd.read_csv("coordinates/coordinates-best.csv")
    worst = pd.read_csv("coordinates/coordinates-worst.csv")
    big = pd.read_csv("coordinates/coordinates-big.csv")

    same = pd.merge(best, worst, on=['name'], how='left')
    same = same.dropna()

    for index, row in best.iterrows():
        for i in range(len(same)):
            if row['name'] == same.iloc[i]['name']:
                best = best.drop(index)

    for index, row in worst.iterrows():
        for i in range(len(same)):
            if row['name'] == same.iloc[i]['name']:
                worst = worst.drop(index)

    plt.scatter(best['lon'], best['lat'], label="Best", marker='o', color="green")
    plt.scatter(worst['lon'], worst['lat'], label="Worst", marker='o', color="red")
    plt.scatter(big['lon'], big['lat'], label="Big", marker='o', color="black")
    # plt.scatter(same['lon_y'], same['lat_y'], label="Uncertain", marker='o', color="orange")

    plt.legend()
    plt.show()


# векторное представление примеров на основе изменения факторов к предыдущему году
def vectorforprev(onetimechange):
    vector = []
    tmp = []
    for i in range(onetimechange.shape[0] - 2):
        if onetimechange[i, 0] == onetimechange[i + 1, 0] == onetimechange[i + 2, 0]:
            vector.append([''] * 20)
            vector.append(np.append(onetimechange[i, :4], [0] * 16))
            for k in range(2):
                for j in range(4, onetimechange.shape[1]):
                    if onetimechange[i + k, j] == onetimechange[i + k + 1, j]:
                        tmp.append(0)
                    else:
                        if onetimechange[i + k, j] < onetimechange[i + k + 1, j]:
                            tmp.append(1)
                        else:
                            tmp.append(-1)
                vector.append(np.append(onetimechange[i + k + 1, :4], tmp))
                tmp = []

    return vector


# векторное представление примеров на основе изменения факторов от переходного года
def vectorforcenter(onetimechange):
    vector = []
    tmp = []
    for i in range(onetimechange.shape[0] - 2):
        if onetimechange[i, 0] == onetimechange[i + 1, 0] == onetimechange[i + 2, 0]:
            vector.append([''] * 20)
            for k in range(3):
                if k == 1:
                    vector.append(np.append(onetimechange[i + 1, :4], [0] * 16))
                else:
                    for j in range(4, onetimechange.shape[1]):
                        if onetimechange[i + 1, j] == onetimechange[i + k, j]:
                            tmp.append(0)
                        else:
                            if onetimechange[i + 1, j] < onetimechange[i + k, j]:
                                tmp.append(1)
                            else:
                                tmp.append(-1)
                    vector.append(np.append(onetimechange[i + k, :4], tmp))
                    tmp = []

    return vector


# векторное представление примеров на основе изменения факторов от переходного года (процентная доля изменения)
def vectorforcenterprop(onetimechange):
    vector = []
    tmp = []
    for i in range(onetimechange.shape[0] - 2):
        if onetimechange[i, 0] == onetimechange[i + 1, 0] == onetimechange[i + 2, 0]:
            vector.append([''] * 20)
            for k in range(3):
                if k == 1:
                    vector.append(np.append(onetimechange[i + 1, :4], [0] * 16))
                else:
                    for j in range(4, onetimechange.shape[1]):
                        if onetimechange[i + 1, j] == onetimechange[i + k, j]:
                            tmp.append(0)
                        else:
                            tmp.append(float(onetimechange[i + k, j] / onetimechange[i + 1, j]) - 1)

                    vector.append(np.append(onetimechange[i + k, :4], tmp))
                    tmp = []

    return vector


# анализ движения поселений между кластерами (отслеживание факторного изменения)
def movementanalyzer(data, clusts):
    notsame = 0
    onetimechange = []
    changeindex = 0
    data = data.sort_values(by=['oktmo', 'year'])

    data = data[data.columns.drop('x')]
    data = data[data.columns.drop('y')]

    minindex = getnegative(clusts)

    for i in range(len(data) - 1):  # если кластер i и i+1 не совпдатает, тогда notsame +1
        if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
            if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
                notsame += 1
                if notsame - 1 == 0:
                    changeindex = i + 1
        else:
            # перешёл в другой кластер и вернулся обратно
            if notsame == 2 and data.iloc[changeindex - 1]['clust'] == data.iloc[changeindex + 1]['clust'] and \
                    data.iloc[changeindex]['clust'] == minindex:
                onetimechange.append(data.iloc[changeindex - 1])
                onetimechange.append(data.iloc[changeindex])
                onetimechange.append(data.iloc[changeindex + 1])

            changeindex = 0
            notsame = 0

    onetimechange = np.array(onetimechange)

    vector = vectorforcenterprop(onetimechange)

    vector = np.array(vector)
    vector = pd.DataFrame(vector, columns=data.columns)
    vector.to_excel("vector of movement-4.xlsx", index=False)


# сохранить данные по всем кластерам в эксель файле (без нормализации)
def saveallclusters(clusts):
    norm = pd.read_csv("datasets/fornorm 24 alltime-clust (IQR)-normbysoul-f.csv")

    writer = pd.ExcelWriter("Clusters.xlsx")
    for k in range(len(clusts)):
        for col in norm:
            clusts[k][col] = clusts[k][col] * norm.iloc[0][col]

        clusts[k] = clusts[k].sort_values(by=['oktmo', 'year'])
        clusts[k].to_excel(writer, sheet_name="Cluster " + str(k) + "", index=False)

    writer.close()


# евклидова метрика
def euclidean(x, y):
    d = 0.0
    for i in range(len(x)):
        d += (x[i] - y[i]) ** 2
    return np.sqrt(d)


# нормирование факторов на душу населения
def normpersoul(tonorm):
    # факторы для нормирования
    features = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool', 'factoriescap']

    for a in features:
        tonorm[a] = float(tonorm[a] / tonorm['popsize'])

    return tonorm


# нормирование факторов на душу населения для всего датасета
def normpersoulalldata(data):
    # факторы для нормирования
    features = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool']

    for a in features:
        data[a] = data[a] / data['popsize']


#демонстрация соц-экономической разницы между двумя МО
def showdifference(worst, best, worstname, bestname):
    dif = []
    #worst = normpersoul(worst)
    #best = normpersoul(best)

    for a in worst.index:
        dif.append(float(best[a] / worst[a]))

    dif = np.array(dif)
    features = list(worst.index)
    dif = pd.Series(dif, index=features)

    #dif = dif.transpose()
    ax = dif.plot.barh()
    ax.set_title("Сравнение " + worstname + " относительно "+ bestname +"")
    ax.set_xlabel('Во сколько раз необходимо улучшить')
    ax.set_ylabel('Социально-экономические индикаторы')
    plt.show()


#демонстрация соц-экономических показателей наиболее близкого МО из лучшего кластера
def headtohead(worst, best, worstname, bestname):
    features = list(worst.index)

    width = 0.3
    x = np.arange(len(features))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, worst, width, label=worstname)
    rects2 = ax.bar(x + width / 2, best, width, label=bestname)
    ax.set_title('Сравнение социально-экономических индикаторов')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    plt.show()


# ретроспективный анализ (развитие наиболее похожего региона)
def retroanalysis(data):
    index = 0
    for i in range(len(data)):
        if 2022 - data.iloc[i]['year'] == 5:
            index = i
            break

    retrodata = data[data['oktmo'] == data.iloc[index]['oktmo']]
    retrodata = retrodata[retrodata['year'] > 2016]
    retrodata = retrodata.sort_values(by='year')
    plt.plot(retrodata['year'], retrodata.iloc[:, 6:21])
    plt.title("График развития соц-экономической среды в "+ retrodata.iloc[0]['name'] +"")
    plt.legend(retrodata.iloc[:, 6:21].columns)
    plt.show()

    print('ok')


# нахождение наиболее похожих МО в кластере согласно социально-экономическим факторам
def siblingsfinder(data, clusts):

    #нормализация набора данных на душу населения
    #normpersoulalldata(data)

    agestruct = pd.read_csv("C:/Users/Albert/.spyder-py3/ITMO-2/migforecasting/superdataset/pop data/agestruct prop.csv")

    # добавление среднего возрастного профиля в сет
    avgagestruct = pd.read_excel('avg agestruct.xlsx')
    agestruct = pd.concat([agestruct, avgagestruct])
    #agestruct = agestruct[agestruct.columns.drop('name')]

    # одинаковые примеры в разных датафреймах (в одном датафрейме дублируются социо-экон. факторы)
    agestruct = pd.merge(data[['oktmo', 'year']], agestruct, how='inner', on=['oktmo', 'year'])
    sync = agestruct[['oktmo', 'year']].drop_duplicates()
    data = pd.merge(sync, data, how='inner', on=['oktmo', 'year'])

    #data = data.drop(columns=['x', 'y'])

    # тавр 52653000
    # спас 29634000
    # домб 53617000
    # тихв 41645000 (14-18)
    # мичу 68715000 (14-19)
    # в демонстративных целях
    # социальные-конфликты
    # 11635000 - (мусорный конфликт 2018, Шиес, Ленский МР)
    # 3623000 - (мусорный конфликт 2023 (2022), ст. Полтавская, Красноармейский МР)
    # 65722000 - (мусорный конфликт 2023 (2022), Сысерть, Сысертский)
    # 63637000 - (межнац. конфликт 2013 (2014), г. Пугачёв, Пугачёвский МР)
    # 56613000 - (межнац. конфликт 2019, с. Чемодановка, Бессоновский МР)
    # 18614000 - (межнац. конфликт 2024 (2017), с. Иловля, Иловлинский МР)
    # 38620000 - (антипромыш. конфликт 2023 (2020), г. Курск, Куриский МР)
    # 80631000 - (антипромыш. конфликт 2020, Куштау, Ишимбайский МР)
    # 80601000 - (антипромыш. конфликт 2020, Крыктытау, Абзелиловский МР)

    index = 0
    for i in range(len(data)):
        if data.iloc[i]['year'] == 2 and data.iloc[i]['oktmo'] == 2:
            index = i
            break

    #index = random.randint(0, len(data))
    # наиболее близкие среди всех кластеров согласно социально-экономическим индикаторам
    sim1 = []
    tmp1 = 0.0
    for b in range(len(data)):
        tmp1 = mean_squared_error(data.iloc[b][6:], data.iloc[index][6:])  # All factors
        sim1.append(tmp1)

    # наиболее близкие среди всех кластеров согласно половозрастной структуре
    sim2 = []
    female = 0.0
    male = 0.0
    popindex = index * 2
    b = 0
    while b < len(agestruct):
        female = mean_squared_error(agestruct.iloc[b][4:18], agestruct.iloc[popindex][4:18])
        male = mean_squared_error(agestruct.iloc[b + 1][4:18], agestruct.iloc[popindex + 1][4:18])
        sim2.append(female + male)
        b+=2

    # добавление критериев в таблицу
    data['similarity 1'] = sim1
    data['similarity 2'] = sim2

    # нормализация оценки подобия от 0 до 1
    data['similarity 1'] = data['similarity 1'] / data['similarity 1'].max()
    data['similarity 2'] = data['similarity 2'] / data['similarity 2'].max()

    # комбинирование двух критериев
    sim3 = (data['similarity 1']) + (data['similarity 2'] * 0.5)
    data['similarity 3'] = sim3

    cols = ['oktmo', 'year', 'name', 'clust', 'similarity 1', 'similarity 2', 'similarity 3', 'saldo', 'popsize',
            'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover', 'livarea', 'sportsvenue',
            'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool', 'factoriescap']

    data = data[cols]

    data = data.sort_values(by='similarity 1')
    data = data.sort_values(by='similarity 2')
    data = data.sort_values(by='similarity 3')

    data = data.reset_index(drop=True)
    #retroanalysis(data)

    # наиболее близкие из лучшего кластера
    norm = pd.read_csv("datasets/fornorm only mundist-f (IQR).csv")
    migprop = 0.0
    bestcluster = 0
    # определение лучшего кластера
    for k in range(len(clusts)):
        msaldo = clusts[k]['saldo'].median() * norm.iloc[0]['saldo']
        mpopsize = clusts[k]['popsize'].median() * norm.iloc[0]['popsize']
        if k == 0:
            migprop = float(msaldo / mpopsize)
        else:
            if migprop < float(msaldo / mpopsize):
                migprop = float(msaldo / mpopsize)
                bestcluster = k

    dist2 = []
    tmp2 = 0.0
    for b in range(len(data)):
        if data.iloc[b]['clust'] == bestcluster or b == 0:
            tmp2 = mean_squared_error(data.iloc[b][6:21], data.iloc[0][6:21])  # All factors
            dist2.append(tmp2)
        else:
            dist2.append(np.NAN)

    data['dist2'] = dist2
    data = data.sort_values(by='dist2')

    # визуализация разницы
    #showdifference(data.iloc[0][6:21], data.iloc[1][6:21], data.iloc[0]['name'], data.iloc[1]['name'])
    headtohead(data.iloc[0][6:21], data.iloc[1][6:21], data.iloc[0]['name'], data.iloc[1]['name'])
    retroanalysis(data)

    # наиболее близкие в своем кластере
    onecluster = clusts[0]
    dist = []
    tmp = 0.0
    for a in range(len(onecluster)):
        tmp = euclidean([onecluster.iloc[a]['x'], onecluster.iloc[a]['y']],
                        [onecluster.iloc[0]['x'], onecluster.iloc[0]['y']])
        dist.append(tmp)

    onecluster['dist'] = dist
    onecluster = onecluster.sort_values(by='dist')
    print('done')


# анализ факторов в кластере (медиана, макс, мин)
def clustsfeatures(clusts, centroids):
    norm = pd.read_csv("datasets/fornorm 24 alltime-clust only mundist (IQR)-normbysoul-f.csv")
    """
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[1]):
            centroids[i, j] = centroids[i, j] * norm.iloc[0, j + 1]

    centroids = np.array(centroids)
    features = list(norm.columns)
    centroids = pd.DataFrame(centroids, columns=features[1:])
    centroids.to_excel("centroids.xlsx", index=False)
    """
    final = []
    tmp = []
    for k in range(len(clusts)):
        tmp.append(k)
        for col in norm:
            tmp.append(clusts[k][col].median() * norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final.append([''] * (norm.shape[1] + 1))
    for k in range(len(clusts)):
        tmp.append(k)
        for col in norm:
            tmp.append(clusts[k][col].max() * norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final.append([''] * (norm.shape[1] + 1))
    for k in range(len(clusts)):
        tmp.append(k)
        for col in norm:
            tmp.append(clusts[k][col].min() * norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final = np.array(final)
    features = list(norm.columns)
    features.insert(0, 'clust')
    final = pd.DataFrame(final, columns=features)
    final.to_excel("median of clusters (normbysoul) only mundist-f.xlsx", index=False)


# анализ кластеров по временному периоду
def analyzer(data, clusts):
    minyear = data['year'].min()
    maxyear = data['year'].max()
    data = data.sort_values(by=['oktmo', 'year'])
    notsame = 0
    solid = 0
    unic_towns = len(pd.unique(data['oktmo']))

    # вычисление кол-ва поселений, которые не меняют свой кластер за временной промежуток
    for i in range(len(data) - 1):
        if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
            if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
                notsame += 1
        else:
            if notsame == 0:
                solid += 1
            else:
                notsame = 0

    # вычисление кол-во данных за конкретный год в кластере
    cluster_year = []
    tmp = []
    for k in range(len(clusts)):
        tmp = []
        for y in range(minyear, maxyear + 1):
            tmp.append(len(clusts[k][clusts[k]['year'] == y]))
        cluster_year.append(tmp)

    cluster_year = np.array(cluster_year)
    cluster_year = pd.DataFrame(cluster_year)
    cluster_year.to_excel("years in clusters.xlsx")

    print('ok')


# получение номера кластера с наименьшим количеством отрицательных сальдо
def getnegative(clusts):
    negprop = []
    for i in range(len(clusts)):
        negprop.append(len(clusts[i][clusts[i]['saldo'] < 0]) / len(clusts[i]))
        plt.bar(i, negprop[i], width=0.3, label="Cluster " + str(i) + "")

    return negprop.index(min(negprop))


# доля поселений с отрицательным сальдо в класетер
def negativeanalyzer(clusts):
    negprop = []
    fig, ax = plt.subplots()
    for i in range(len(clusts)):
        negprop.append(len(clusts[i][clusts[i]['saldo'] < 0]) / len(clusts[i]))

    y = range(0, len(clusts))
    ax.bar(y, negprop, width=0.3, color=sns.color_palette('viridis', len(clusts)))

    plt.title("Proportion of settlements with negative net migration")
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.show()
    """
    # выгрузка уникальных поселений из наиболее и наименее отрицательного кластера (согласно сальдо)
    minindex = negprop.index(min(negprop))
    maxindex = negprop.index(max(negprop))

    bestcities = clusts[minindex]['name'].unique()
    bestcities = np.sort(bestcities)
    bestcities = bestcities.reshape(-1, 1)

    bestcities = pd.DataFrame(bestcities)
    bestcities.to_csv("bestcities.csv", index=False)

    worstcities = clusts[maxindex]['name'].unique()
    worstcities = np.sort(worstcities)
    worstcities = worstcities.reshape(-1, 1)

    worstcities = pd.DataFrame(worstcities)
    worstcities.to_csv("worstcities.csv", index=False)
    """


# медианное значение сальдо в кластере
def getmedian(data2):
    sns.boxplot(x='clust', y='saldo', data=data2, palette='viridis')

    plt.title("Median values of net migration")
    plt.xlabel('Cluster')
    plt.ylabel('Net migration')
    plt.show()


# оценка значимости через классификатор
def findsignif(data2):
    y = data2['clust']
    y = y / (k - 1)  # нормализация

    data2 = data2[data2.columns.drop('clust')]

    class_model = RandomForestRegressor(n_estimators=100, random_state=0)
    class_model.fit(data2, y)

    explainer = shap.TreeExplainer(class_model)
    shap_values = explainer(data2).values

    shap.summary_plot(shap_values, data2)


# метод Антона Николаевича (средние показатели индикаторов для отрицательных и положительных примеров в кластере)
def ANmethod(clusts):
    norm = pd.read_csv("datasets/fornorm 24 alltime-clust (IQR)-normbysoul-f.csv")
    final = []
    tmpnegative = []
    tmppositive = []
    # вычисление средних значений и нормализация
    for k in range(len(clusts)):
        negative = clusts[k][clusts[k]['saldo'] < 0]
        positive = clusts[k][clusts[k]['saldo'] > 0]

        #negative = negative.drop(columns=['x', 'y'])
        #positive = positive.drop(columns=['x', 'y'])

        tmpnegative.append(k)
        tmppositive.append(k)
        for col in negative:
            if col != 'oktmo' and col != 'name' and col != 'year' and col != 'clust':
                tmppositive.append(positive[col].mean() * norm.iloc[0][col])
                tmpnegative.append(negative[col].mean() * norm.iloc[0][col])
        final.append(np.array(tmppositive))
        final.append(np.array(tmpnegative))
        tmppositive = []
        tmpnegative = []
        final.append([''] * len(final[0]))

    final = np.array(final)
    features = list(norm.columns)
    features.insert(0, 'clust')
    final = pd.DataFrame(final, columns=features)
    final.to_excel("AN-method output (all).xlsx", index=False)


k = 6  # кол-во кластеров

data = pd.read_csv("datasets/superdataset-24 alltime-clust (IQR)-normbysoul-f.csv")

profile = pd.read_excel('avg profile.xlsx')

# нормализация среднего профиля и добавление в сет
norm = pd.read_csv("datasets/fornorm 24 alltime-clust (IQR)-normbysoul-f.csv")
for i in range(len(profile)):
    for j in range(5, profile.shape[1]):
        profile.iloc[i, j] = profile.iloc[i, j] / norm.iloc[0, j - 3]

data = pd.concat([data, profile])

num = data['oktmo'].nunique()

#normpersoulalldata(data)

data = data.sample(frac=1)  # перетасовка

# модель кластеризации
clust_model = KMeans(n_clusters=k, random_state=None, n_init='auto')
clust_model.fit(data.iloc[:, 5:])  # 4 - без сальдо, 5 - без popsize (при нормировании на душу)

# clust_model = AgglomerativeClustering(n_clusters=k, linkage='ward')
# clust_model.fit_predict(data.iloc[:, 4:])

#print(silhouette_score(data.iloc[:, 5:], clust_model.labels_, metric='euclidean'))
print(clust_model.inertia_)

centroids = clust_model.cluster_centers_

# сохранение модели
#joblib.dump(clust_model, 'kmeans_model (24-all-iqr-normbysoul).joblib')

# добавляем к данным столбец с номером кластера
data['clust'] = clust_model.labels_

cols = ['oktmo', 'year', 'name', 'clust', 'saldo', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats',
        'retailturnover', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
        'livestock', 'harvest', 'agrprod', 'hospitals', 'beforeschool', 'factoriescap']

data = data[cols]

data = data.sort_values(by=['oktmo', 'year'])

#data.to_csv("For Sergey (all).csv", index=False)

# трансформация в 2D методом компонент
#pca = PCA(2)
#pca2 = pca.fit_transform(data.iloc[:, 6:])  # 5 - без сальдо, 6 - без popsize
#data['x'] = pca2[:, 0]
#data['y'] = pca2[:, 1]

# разделяем кластеры по независимым массивам (массив массивов)
clusts = []
for i in range(k):
    clusts.append(data[data['clust'] == i])

# анализ и вывод результатов

#ANmethod(clusts)

siblingsfinder(data, clusts)

getmedian(data)

negativeanalyzer(clusts)

#clustsfeatures(clusts, centroids)

#saveallclusters(clusts)

# saveallclusters(clusts)

# movementanalyzer(data, clusts)

# analyzer(data, clusts)

# getnegative(clusts)

# findsignif(data)

x = [1, 2, 3, 1, 2, 3]
y = [2, 3, 2, -2, -3, -2]

for i in range(k):
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "")

plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

for i in range(k):
    clusts[i]['x'] = x[i]
    clusts[i]['y'] = y[i]
    plt.scatter(clusts[i]['x'], clusts[i]['y'], label="Cluster " + str(i) + "", marker='o', s=160)

data = data.sort_values(by=['oktmo', 'year'])

# вычисление количества перемещений в конкретном направлении между кластерами
relation = {}
for i in range(int(len(data) - 1)):
    if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
        if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
            try:
                relation["" + str(int(data.iloc[i]['clust'])) + "-" + str(int(data.iloc[i + 1]['clust'])) + ""] += 1
            except KeyError:
                relation["" + str(int(data.iloc[i]['clust'])) + "-" + str(int(data.iloc[i + 1]['clust'])) + ""] = 1

relation = dict(sorted(relation.items(), key=lambda item: item[1], reverse=True))

for key in relation:
    cor = key.split('-')
    plt.plot((x[int(cor[0])], x[int(cor[1])]), (y[int(cor[0])], y[int(cor[1])]),
             linewidth=(relation[key] * 0.03), color="black")

"""
count = 0
for i in range(int(len(data) - 1)):
    if data.iloc[i]['oktmo'] == data.iloc[i + 1]['oktmo']:
        if data.iloc[i]['clust'] != data.iloc[i + 1]['clust']:
            #plt.plot(data.iloc[i:i+2]['x'], data.iloc[i:i+2]['y'], color='grey')
            plt.plot((x[int(data.iloc[i]['clust'])], x[int(data.iloc[i + 1]['clust'])]),
                     (y[int(data.iloc[i]['clust'])], y[int(data.iloc[i + 1]['clust'])]), color='grey')
            count+=1
"""

plt.legend()
plt.title("Разбиение данных на " + str(k) + " кластера")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print('done')
