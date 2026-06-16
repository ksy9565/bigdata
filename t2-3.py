"""
성인 인구조사 소득 예측
- age: 나이
- workclass: 고용 형태
- fnlwgt: 사람의 대표성을 나타내는 가중치(final weight)
- education: 교육 수준
- education.num: 교육 수준 수치
- marital.status: 결혼 상태
- occupation: 업종
- relationship: 가족 관계
- race: 인종
- sex: 성별
- capital.gain: 양도 소득
- capital.loss: 양도 손실
- hours.per.week: 주당 근무 시간
- native.country: 국적
- income: 수익 (예측해야 하는 값)
"""
# %% 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


df = pd.read_csv("archive/adult.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='income', null_name='?')

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
X_train.head()
# %%
y_train['income'].value_counts()
# %%
X_train.info()
# %%
numeric_feat = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categoric_feat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                  'native.country']
# %%
X_train[numeric_feat].describe()
# %%
X_train.isnull().sum()
# %%
X_test.isnull().sum()
# %%
X_train['workclass'].value_counts().sum()


# %%
def data_fillna(df):
    df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
    df['occupation'] = df['occupation'].fillna("null")
    df['native.country'] = df['native.country'].fillna(df['native.country'].mode()[0])
    return df


X_train = data_fillna(X_train)
X_test = data_fillna(X_test)
# %%
X_train.isnull().sum()
# %%
X_test.isnull().sum()
# %%
from sklearn.preprocessing import LabelEncoder

all_df = pd.concat([X_train.assign(ind="train"), X_test.assign(ind="test")])
le = LabelEncoder()
all_df[categoric_feat] = all_df[categoric_feat].apply(le.fit_transform)

X_train = all_df[all_df['ind'] == 'train']
X_train = X_train.drop('ind', axis=1)
# %%
X_test = all_df[all_df['ind'] == 'test']
X_test = X_test.drop('ind', axis=1)
# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train[numeric_feat] = scaler.fit_transform(X_train[numeric_feat])
X_test[numeric_feat] = scaler.fit_transform(X_test[numeric_feat])
# %%
y = (y_train['income'] != '<=50K').astype(int)
# %%
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.15, random_state=0)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
# %%
X_tr = X_tr.drop('id', axis=1)
X_val = X_val.drop('id', axis=1)
# %%
X_tr.head(1)
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=0)
model.fit(X_tr, y_tr)
pred = model.predict(X_val)
print('accuracy score:', accuracy_score(y_val, pred))
# %%
X_test_id = X_test.pop('id')
pred = model.predict(X_test)
# %%
output = pd.DataFrame({'id': X_test_id, 'income': pred})
output.to_csv("t2-3.csv", index=False)
