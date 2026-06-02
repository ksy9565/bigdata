# %%
"""
문제 2. [직원 퇴사 예측 분석]
제공 파일: employee_attrition.csv
설명: 해당 데이터는 직원의 만족도, 급여 수준, 프로젝트 개수, 주간 근무 시간 등과
실제 퇴사 여부 정보를 담고 있습니다.
이를 바탕으로 퇴사 확률을 예측하는 로지스틱 회귀 분석을 수행하세요.

공통 지시사항
- 독립변수: satisfaction, salary_level, project_count, working_hours
- 종속변수: left_company (퇴사 여부: 0 = 재직, 1 = 퇴사)
- salary_level은 범주형 변수이며, 기준 수준(reference level)은 low로 설정합니다.
"""
import pandas as pd

df = pd.read_csv("archive/employee_attrition.csv")
df_head = df.head()

from statsmodels.formula.api import logit

model = logit(
    'left_company ~ satisfaction + C(salary_level, Treatment(reference="low")) + project_count + working_hours',
    data=df).fit()
# %%
"""
문제 2-1
로지스틱 회귀모형을 적합한 후,
satisfaction 변수의 회귀계수를
소수 셋째 자리까지 반올림하여 구하시오.
"""
print('회귀계수 :', model.summary())

# 코드로 답 찾기
satisfaction_coef = round(model.params['satisfaction'], 3)  # -2.844
# %%
"""
문제 2-2 (심화 문제)
회귀모형 결과를 바탕으로,
salary_level = high인 직원이
low인 직원에 비해 퇴사할 오즈비(Odds Ratio)를 계산하시오.
결과는 소수 셋째 자리까지 반올림하여 제시하시오.
"""
import numpy as np

odds_ratio = round(np.exp(model.params['C(salary_level, Treatment(reference="low"))[T.high]']), 3)
print('오즈비: ', odds_ratio)  # 0.099
# %%
"""
문제 2-3
적합한 로지스틱 회귀모형을 이용하여
각 직원의 퇴사 확률을 예측한 후,
예측 확률이 0.5 이상인 직원 중
실제로 퇴사한 직원의 비율(정밀도, precision)을 계산하시오.
결과는 소수 셋째 자리까지 반올림하여 구하시오.
"""
from sklearn.metrics import precision_score

pred_prob = model.predict(df)
pred_label = (pred_prob >= 0.5).astype(int)
precision_val = round(precision_score(df['left_company'], pred_label), 3)
print('정밀도: ', precision_val)
