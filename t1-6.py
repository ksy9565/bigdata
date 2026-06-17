"""
결측치 제거 및 그룹 합계에서 조건에 맞는 값 찾아 출력
주어진 데이터 중 basic1.csv에서 'f1'컬럼 결측 데이터를 제거하고,
'city'와 'f2'을 기준으로 묶어 합계를 구하고,
'city가 경기이면서 f2가 0'인 조건에 만족하는 f1 값을 구하시오
"""
# 라이브러리 및 데이터 불러오기
# f1컬럼 결측치 제거
# 그룹 합계 계산
# 조건에 맞는 값 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %% 'f1'컬럼 결측 데이터 제거
df = df.dropna(subset=['f1'])
# %%
df.head()
# %% 'city'와 'f2'을 기준으로 묶어 합계 구함
df2 = df.groupby(['city', 'f2']).sum()
# %%
df2.iloc[:, 1:]
# %% 'city'가 경기이면서 'f2'가 0인 조건에 만족하는 f1 값
print(df2.iloc[0]['f1'])
