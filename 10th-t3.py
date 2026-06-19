"""
문제1.
제공된 데이터(attrition.csv)는
직원의 나이, 연봉, 근무 특성 등에 따른
이직 여부(attrition)에 관한 자료이다.
이 데이터를 이용하여 로지스틱 회귀분석(Logistic Regression)을 수행하고,
제시된 소문항에 답하시오.

[제공 데이터]
- attrition : 이직 여부 (0 = 잔류, 1 = 이직) 종속변수
- age : 나이 (정수형)
- income : 연봉 (정수형)
- overtime : 초과근무 여부 (0 = 아니오, 1 = 예)

문제1-1.
로지스틱 회귀모델을 적합한 후,
p-value가 0.05보다 작은 변수의 회귀계수를 구하시오.
단 절편항(상수항)은 제외한다. (반올림하여 소수 셋째 자리까지 작성)
"""
import pandas as pd

df = pd.read_csv('archive/attrition.csv')

# 로지스틱 회귀분석 진행
from statsmodels.formula.api import logit

model = logit("attrition ~ age + income + overtime", data=df).fit()

# 회귀분석 결과 출력
print(model.summary())

coef = model.params['income']
print(round(coef, 3))
"""
문제1-2.
나이(age)가 1 증가할 때 이직할 오즈비(Odds Ratio)를 구하시오.
(반올림하여 소수 셋째 자리까지 작성)
"""
import numpy as np

print(round(np.exp(model.params['age']), 3))
"""
문제1-3.
새로운 직원(age=40, income=4500, overtime=1)에 대해,
모델을 이용하여 이직 확률을 예측하시오.
(반올림하여 소수 셋째 자리까지 작성)
"""

new = pd.DataFrame({'age': [40], 'income': [4500], 'overtime': [1]})
pred = model.predict(new)
print(round(pred[0], 3))
