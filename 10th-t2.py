"""
문제¶
제공된 학습용 데이터(train.csv)는
서울시 각 시군구별 건물 및 시설 현황 데이터이다.
학습용 데이터를 활용하여 총가스사용량을 예측하는 모형을 개발하고,
모형을 평가용 데이터(test.csv)에 적용하여 총가스사용량을 예측하시오.

예측 결과는 아래의 [유의 사항]과 [제출 형식]을 준수하여,
CSV 파일(result.csv)을 생성하는 코드를 제출하시오.

[유의 사항]
① 예측 결과는 RMSE(Root Mean Squared Error) 평가 지표에 따라 평가함
② 타겟 변수(총가스사용량)는 일부 값이 0으로 기재되어 있으며,
   이는 결측치를 대체한 값임.

[제출 형식]
① CSV 파일명: result.csv
② 예측 총가스사용량 컬럼명: pred
③ 제출 컬럼 개수: pred 컬럼 1개
④ 평가용 데이터 개수와 예측 결과 데이터 개수 일치: 1,476개

[제공 데이터]
train.csv: 학습용 데이터, 약 3,196개
test.csv: 평가용 데이터, 약 1,476개 평가용 데이터에는 총가스사용량 컬럼 미제공

[CSV 파일 형식 및 확인 방법]
CSV 파일명: result.csv
예측 총가스사용량 컬럼명: pred
제출 CSV 파일 예시
```
pred
8790
14748
21560
761
```
"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.unicode.east_asian_width', True)

train = pd.read_csv('archive/gas_train.csv')
test = pd.read_csv('archive/gas_test.csv')

# EDA
print(train.info())
print(train.head())
print(train.describe())
print(train.shape)

cond = train['총가스사용량'] != 0
train = train[cond]

print(train.describe())
print(train.shape)

print(train.isnull().sum())
print(test.isnull().sum())

# 전처리
from sklearn.preprocessing import LabelEncoder

target = train.pop('총가스사용량')

df = pd.concat([train, test], ignore_index=True)
le = LabelEncoder()
df['시군구명'] = le.fit_transform(df['시군구명'])

X_train = df.iloc[:len(train)].reset_index(drop=True)
test = df.iloc[len(train):].reset_index(drop=True)

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.2, random_state=0)

# model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0)
model.fit(X_tr, y_tr)
pred = model.predict(X_val)

# 평가
from sklearn.metrics import root_mean_squared_error, r2_score

print("RMSE: ", root_mean_squared_error(y_val, pred))
print("r2: ", r2_score(y_val, pred))

pred = model.predict(test)
submit = pd.DataFrame({'pred': pred})
# submit.to_csv('10th-t2.csv', index=False)
print(submit.head())
print(submit.shape)
