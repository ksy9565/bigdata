"""
집 값 예측
예측할 변수 ['SalePrice']
평가: rmse, r2
- rmse는 낮을 수록 좋은 성능
- r2는 높을 수록 좋은 성능
"""
from tabnanny import verbose

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name

    if null_name != "":
        df[df == null_name] = np.nan

    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test


df = pd.read_csv("archive/t2-4/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%
X_train.head(5)
# %%
y_train.head(5)
# %%
import matplotlib.pyplot as plt

y_train['SalePrice'].hist()
plt.show()

y_test['SalePrice'].hist()
plt.show()
# %%
X_train.isnull().sum().sort_values(ascending=False)[:20]
# %%
X_train = X_train.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
target = y_train['SalePrice']
# %%
X_train.shape
# %%
X_train.isnull().sum()
# %%
X_train['LotFrontage'].value_counts()
# %%
from sklearn.impute import SimpleImputer

imp = SimpleImputer()
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
# %%
X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=0)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


model = RandomForestRegressor()
model.fit(X_tr, y_tr)
pred = model.predict(X_val)

print("R2: " + str(r2_score(y_val, pred)))
print("RMSE: " + str(rmse(y_val, pred)))
# %%
y_test.mean()
