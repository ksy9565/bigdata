"""
주어진 데이터는 고혈압 환자 치료 전후의 혈압이다.
해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오
귀무가설(H0): μ >= 0
대립가설(H1): μ < 0
μ = (치료 후 혈압 - 치료 전 혈압)의 평균
유의수준: 0.05

1. μ의 표본평균은?(소수 둘째자리까지 반올림)
2. 검정통계량 값은?(소수 넷째자리까지 반올림)
3. p-값은?(소수 넷째자리까지 반올림)
4. 가설검정의 결과는? (유의수준 5%)
"""
# %%
import pandas as pd

df = pd.read_csv("archive/high_blood_pressure.csv")
# %% 대응 표본 t-검정
from scipy.stats import ttest_rel

df['diff'] = df['bp_post'] - df['bp_pre']
print(round(df['diff'].mean(), 2))

t_statistic, p_value = ttest_rel(df['bp_post'], df['bp_pre'], alternative="less")

print("검정통계랑:", t_statistic)
print("p-value:", p_value)
alpha = 0.05

if p_value < alpha:
    print("귀무가설 기각. 치료 효과 있음")
else:
    print("귀무가설 채택. 치료 효과 없음")
