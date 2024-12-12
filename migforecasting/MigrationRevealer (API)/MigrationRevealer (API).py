import pandas as pd

from fastapi import FastAPI
from fastapi import Body
from fastapi import Request
import uvicorn

app = FastAPI()


#Нормирование рублевых цен
def normbyinf(inputdata):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    infdata = pd.read_csv("inflation14.csv")
    for k in range(len(inputdata)):
        inflation = infdata[infdata['year'] == inputdata.iloc[k]['year']]   # получить инфляцию за необходимый год
        for col in thisrubfeatures:
            index = inputdata.columns.get_loc(col)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * (inflation.iloc[0]['inf'] / 100)

    return inputdata


# Нормирование данных для модели
def normformodel(inputdata):
    norm = pd.read_csv("fornorm-24.csv")
    final = []
    tmp = []
    for k in range(len(inputdata)):
        for col in norm:
            if col != 'saldo':
                tmp.append(inputdata.iloc[k][col] / norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final = np.array(final)
    features = list(norm.columns[1:])
    final = pd.DataFrame(final, columns=features)
    inputdata = final
    return inputdata


@app.get("/migration-revealer")
async def reveal(request: Request):
    features = ['year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 'foodseats', 'retailturnover', 'livarea',
                'sportsvenue', 'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 'hospitals',
                'beforeschool', 'factoriescap']

    x = dict(request.query_params)
    x = pd.DataFrame(x, index=[0])
    x = x.transpose()
    x = x[features]     # правильный порядок для модели

    return mig


@app.get("/notsure")
async def calc(year: int, popsize: int, avgemployers: float, avgsalary: float, shoparea: float,
               foodseats: int, retailturnover: float, livarea: int, sportsvenue: int, servicesnum: int, roadslen: float,
               livestock: int, harvest: float, agrprod: float, hospitals: int, beforeschool: int, factoriescap: float):
    return avgsalary


@app.get("/getjson")
async def get_body(request: Request):
    x = dict(request.query_params)
    x = pd.DataFrame(x, index=[0])
    x = x.transpose()
    data = pd.read_json("anyinput_test.json")
    return x


@app.get("/getget")
async def hello(data=Body()):
    return data


if __name__ == "__main__":
    uvicorn.run("MigrationRevealer (API):app", host="0.0.0.0", port=8000, reload=True)
