"""
작업형1 문제
1-1. age 컬럼의 3사분위수와 1사분위수의 차를
절대값으로 구하고, 소수점 버려서, 정수로 출력
"""
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('archive/basic1.csv')

print(df.head())
q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
print(int(abs(q3 - q1)))
"""
1-2.(loves반응+wows반응)/(reactions반응) 비율이
0.4보다 크고 0.5보다 작으면서,
type 컬럼이 'video'인 데이터의 갯수
"""
df = pd.read_csv('archive/4th/fb.csv')

print(df.head())
stm = (df['loves'] + df['wows']) / df['reactions']
cond = df[(stm > 0.4) & (0.5 > stm) & (df['type'] == 'video')]
print(len(cond))
"""
1-3. date_added가 2018년 1월 이면서
country가 United Kingdom 단독 제작인 데이터의 갯수
"""
df = pd.read_csv('archive/4th/nf.csv')

print(df.head())
df['date_added'] = pd.to_datetime(df['date_added'])
print(df.head())

cond1 = (df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1)
cond2 = df['country'] == 'United Kingdom'
print(len(df[cond1 & cond2]))
