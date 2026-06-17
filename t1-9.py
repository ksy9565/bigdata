"""
수치형 변수 표준화
주어진 데이터에서 'f5'컬럼을 표준화
(Standardization (Z-score Normalization))
하고 그 중앙값을 구하시오
"""
# 라이브러리 및 데이터 불러오기
# 표준화
# 중앙값 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %% 표준화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['f5'] = scaler.fit_transform(df[['f5']])
# %%
df.head()
# %% 중앙값 출력
print(df['f5'].median())
