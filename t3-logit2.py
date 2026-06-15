"""
Pclass, Gender, sibsp, parch를 독립변수로 사용하여
로지스틱 회귀모형을 실시하였을 때, parch변수의 계수값은?
단, Pclass는 범주형 변수이다
(반올림하여 소수 셋째 자리까지 계산)
"""
# %%
import pandas as pd

df = pd.read_csv("archive/Titanic.csv")
# %%
from statsmodels.formula.api import logit

formula = "Survived ~ C(Pclass) + Gender + SibSp + Parch"
model = logit(formula, data=df).fit()
print(model.params)
