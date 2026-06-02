# %%
"""
문제 1. [광고 효과 분석]
제공 파일: ad_campaign.csv
설명: 해당 데이터는 광고 캠페인의 예산(ad_tv, ad_radio, ad_social, ad_search, ad_email)과
      결과 매출(sales) 정보를 담고 있습니다.

공통 지시사항
- 전체 데이터는 200개 샘플로 구성되어 있습니다.
- 상위 150개는 학습용 데이터(train), 하위 50개는 검증용 데이터(test)로 사용하세요.
    train = df.iloc[:150]
    test = df.iloc[150:]
"""
import pandas as pd

df = pd.read_csv("archive/ad_campaign.csv")
df_head = df.head()
# %%
"""
문제 1-1
train 데이터에서 ad_tv, ad_radio, ad_social, ad_search, ad_email을 독립변수로,
sales를 종속변수로 하는 다중 선형 회귀모형을 적합하시오.
이때, 회귀계수가 음수인 독립변수의 개수를 구하시오.
"""
from statsmodels.formula.api import ols

# 데이터 분리
train = df.iloc[:150]
test = df.iloc[150:]

# 다중 선형 회귀 적합 후 음수 계수 개수 구하기
model = ols('sales ~ ad_tv + ad_radio + ad_social + ad_search + ad_email', data=train).fit()
print(model.summary())  # ad_social, ad_email

# 코드로 답 찾기
negative_coef_count = (model.params.drop('Intercept') < 0).sum()
print('음수 계수 개수: ', negative_coef_count)
# %%
"""
문제 1-2
1-1에서 적합한 회귀모형의 결과를 바탕으로,
ad_social 변수의 95% 신뢰구간의 하한값을 반올림하여 소수 셋째 자리까지 구하시오.
"""
print(round(model.conf_int(), 3))  # ad_social: -0.075

# 코드로 답 찾기
ad_social_ci_lower = round(model.conf_int().loc['ad_social'][0], 3)
print(ad_social_ci_lower)
# %%
"""
문제 1-3
test 데이터에 대해 위 회귀모형을 사용해 sales를 예측하고,
예측값의 평균을 소수 셋째 자리까지 반올림하여 구하시오.
"""
test_pred_mean = model.predict(test).mean()
print(round(test_pred_mean, 3))
