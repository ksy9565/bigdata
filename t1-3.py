"""
결측치 처리
주어진 데이터에서 결측치가 80% 이상 되는 컬럼은 삭제하고,
80% 미만인 결측치가 있는 컬럼은 'city'별 중앙값으로 값을 대체하고
'f1'컬럼의 평균값을 출력하세요!
"""
# 라이브러리 및 데이터 불러오기
# EDA - 결측값 확인(비율 확인)
# 80%이상 결측치 컬럼, 삭제
# 80%미만 결측치 컬럼, city별 중앙값으로 대체
# f1 평균값 결과 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %%
df.head()
# %%
df.isnull().sum() / df.shape[0]
# %%
df = df.drop(['f3'], axis=1)
print(df.shape)
# %%
df['city'].unique()
seoul = df[df['city'] == '서울']['f1'].median()
busan = df[df['city'] == '부산']['f1'].median()
daegu = df[df['city'] == '대구']['f1'].median()
gyeongi = df[df['city'] == '경기']['f1'].median()
# %%
df.head()
# %%
df['f1'] = df['f1'].fillna(df['city'].map({'서울': seoul, '부산': busan, '대구': daegu, '경기': gyeongi}))
# %%
print(df['f1'].mean())
