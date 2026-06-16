# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name

    if null_name != "":
        df[df == null_name] = np.nan

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)

    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test


df = pd.read_csv("archive/insurance2.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='charges')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %% EDA
X_train.head()
# %%
y_train.head()
# %%
import matplotlib.pyplot as plt

y_train['charges'].hist()
plt.show()
# %%
X_train.isnull().sum()
# %%
X_test.isnull().sum()
# %%
X_train.info()
# %%
X_train.select_dtypes("object").columns
# %%
cols = X_train.select_dtypes("object").columns

for col in cols:
    print("\n=====", col, "=====")
    print("[train]")
    print(X_train[col].value_counts())
    print("[test]")
    print(X_test[col].value_counts())
# %% 전처리
# 범주형 변수
X_train = pd.get_dummies(X_train, columns=cols)
X_test = pd.get_dummies(X_test, columns=cols)
# %%
X_train.head(2)
# %% Log Transform
y_train['charges'].hist()
plt.show()
# %%
y_train['charges'] = np.log1p(y_train['charges'])
y_train['charges'].hist()
plt.show()
# %% Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train['bmi'] = scaler.fit_transform(X_train[['bmi']])
X_test['bmi'] = scaler.fit_transform(X_test[['bmi']])

# %% label encoding
# 나이를 10, 20, 30대로 구분 -> 10으로 나눈 몫
X_train['age'] = X_train['age'].apply(lambda x: x // 10)
X_test['age'] = X_test['age'].apply(lambda x: x // 10)
# %%
X_train.head(3)
# %% Train-Validation Split
target = y_train['charges']
X_train = X_train.drop('id', axis=1)
# %%
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=2022)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_tr, y_tr)
pred = rf.predict(X_val)
# %%
from sklearn.metrics import mean_squared_error


def rmse2(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# %%
rmse2(y_val, pred)


# %%
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# %%
rmse(y_val, pred)
# %%
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_tr, y_tr)
pred = xgb.predict(X_val)
# %%
rmse2(y_val, pred)
# %%
rf.fit(X_train, y_train['charges'])
pred = rf.predict(X_test.drop('id', axis=1))
# %%
pred = np.exp(pred)
output = pd.DataFrame({'id': y_test['id'], 'charges': pred})
# %%
output.head()
# %%
rmse(y_test['charges'], pred)
