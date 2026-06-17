"""
조건에 맞는 데이터 표준편차 구하기
주어진 데이터 중 basic1.csv에서
'f4'컬럼 값이 'ENFJ'와 'INFP'인 'f1'의
표준편차 차이를 절대값으로 구하시오
"""
# 라이브러리 및 데이터 불러오기
# 조건에 맞는 데이터 (ENFJ, INFP)
# 조건에 맞는 f1의 표준편차 (ENFJ, INFP)
# 두 표준편차 차이 절대값 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %%
df.head()
# %%
df = df[df['f4'].isin(['ENFJ', 'INFP'])]
# %%
enfj_std = df[df['f4'] == 'ENFJ']['f1'].std()
# %%
infp_std = df[df['f4'] == 'INFP']['f1'].std()
# %%
print(abs(enfj_std - infp_std))
