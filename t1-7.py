"""
값 변경 및 2개 이상의 조건
'f4'컬럼의 값이 'ESFJ'인 데이터를 'ISFJ'로 대체하고,
'city'가 '경기'이면서 'f4'가 'ISFJ'인 데이터 중
'age'컬럼의 최대값을 출력하시오!
"""
# 라이브러리 및 데이터 불러오기
# ESFJ 값을 가진 데이터 확인
# 값 변경하기
# 2개의 조건에 맞는 값중 age컬럼의 최대값
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %%
df.head()
# %%
df.loc[df['f4'] == 'ENFJ', 'f4'] = 'ISFJ'
# %%
df.iloc[0]
# %%
print(df[(df['city'] == '경기') & (df['f4'] == 'ISFJ')]['age'].max())
