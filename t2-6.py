# %%
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
train = pd.read_csv("archive/bike-regressor/train.csv")
test = pd.read_csv("archive/bike-regressor/test.csv")
# %% EDA
train.shape, test.shape
# %%
train.head(2)
# %%
test.head(2)
# %%
train.isnull().sum()
# %%
test.isnull().sum()
# %%
import matplotlib.pyplot as plt

train['count'].hist()
plt.show()
# %% 전처리
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day

train = train.drop('datetime', axis=1)
test = test.drop('datetime', axis=1)
# %%
train = train.drop(['casual', 'registered'], axis=1)
# %%
train.head(1)
# %%
target = train.pop('count')
# %%
target
# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2023)
# %%
X_train.shape, X_val.shape, y_train.shape, y_val.shape
# %%
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_val)
# %%
r2_score(y_val, pred)
