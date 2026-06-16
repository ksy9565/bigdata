"""
이상치를 찾아라
데이터에서 IQR을 활용해 Fare컬럼의 이상치를 찾고,
이상치 데이터의 여성 수를 구하시오
"""
# 라이브러리 및 데이터 불러오기
# 간단한 탐색적 데이터 분석 (EDA)
# IQR 구하기
# 이상치 데이터 구하기
# 이상치 데이터에서 여성 수 구하기, 출력하기 print()
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/Titanic.csv')
# %%
print(df.shape)
print(df.isnull().sum())
print(df.head())
# %% IQR 구하기
Q1 = df['Fare'].quantile(.25)
Q3 = df['Fare'].quantile(.75)

IQR = Q3 - Q1
# %%
Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
# %%
outdata1 = df[df['Fare'] < (Q1 - 1.5 * IQR)]
outdata2 = df[df['Fare'] > (Q3 + 1.5 * IQR)]
# %%
len(outdata1), len(outdata2)
# %%
print(sum(outdata2['Gender'] == 'female'))
