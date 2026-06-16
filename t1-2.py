"""
이상치를 찾아라(소수점 나이)
주어진 데이터에서 이상치(소수점 나이)를 찾고
올림, 내림, 버림(절사)했을때
3가지 모두 이상치 'age' 평균을 구한 다음
모두 더하여 출력하시오
"""
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %%
df.head()
# %%
df.info()
# %%
df = df[(df['age'] - np.floor(df['age']) != 0)]
# %%
m_ceil = np.ceil(df['age']).mean()
m_floor = np.floor(df['age']).mean()
m_trunc = np.trunc(df['age']).mean()
print(m_ceil + m_floor + m_trunc)
