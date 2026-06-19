"""
Q. 2022년 데이터 중 2022년 중앙값보다 큰 값의 데이터 수
data: t1-data2.csv
"""
import pandas as pd

df = pd.read_csv('archive/big-data-analytics-certification/t1-data2.csv', index_col='year')
print(df.head())
m = df.loc['2022년'].median()
print(m)
cond = df.loc['2022년'] > m
print(df.loc['2022년'][cond].count())
"""
Q. 결측치 데이터(행)을 제거하고,
앞에서부터 60% 데이터만 활용해,
'f1' 컬럼 3사분위 값을 구하시오
60%가 소수점일 경우 절사
(예: 36.6 일때 36으로 계산)

data: t1-data1.csv
"""
import pandas as pd

df = pd.read_csv("archive/big-data-analytics-certification/t1-data1.csv")

print(df.head())
df = df.dropna(axis=0)
print(df.head())
df = df.iloc[:int(len(df) * 0.6)]
q3 = df['f1'].quantile(0.75)
print(q3)
"""
Q. 결측치가 제일 큰 값의 컬럼명을 구하시오
data: t1-data1.csv
"""
import pandas as pd

df = pd.read_csv("archive/big-data-analytics-certification/t1-data1.csv")

print(df.head())
print(df.isnull().sum().sort_values(ascending=False).index[0])
