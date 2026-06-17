"""
누적합 그리고 보간(결측치 처리)
주어진 데이터 셋에서 'f2' 컬럼이 1인 조건에 해당하는
데이터의 'f1'컬럼 누적합을 계산한다.
이때 발생하는 누적합 결측치는 바로 뒤의 값을 채우고,
누적합의 평균값을 출력한다.
(단, 결측치 바로 뒤의 값이 없으면 다음에 나오는 값을 채워넣는다)
"""
# 라이브러리 및 데이터 불러오기
# 조건에 따른 누적합
# 결측치 처리 (뒤에 나오는 값으로 채움)
# 평균 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/basic1.csv')
# %% 조건에 따른 누적합
cumulative = df[df['f2'] == 1]['f1'].cumsum()
# %%
cumulative
# %%
# 결측치 처리 (뒤에 나오는 값으로 채움)
# 평균 출력
print(cumulative.bfill().mean())
