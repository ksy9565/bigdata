"""
왜도와 첨도 구하기
주어진 데이터 중 train.csv에서
'SalePrice'컬럼의 왜도와 첨도를 구한 값과,
'SalePrice'컬럼을 스케일링(log1p)로 변환한 이후
왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오
"""
"""
왜도
왜도는 실수 값 확률 변수의 확률 분포 비대칭성을 나타내는 지표이다.
왜도의 값은 양수나 음수가 될 수 있으며 정의되지 않을 수도 있다.
왜도가 음수일 경우에는 확률밀도함수의 왼쪽 부분에 긴 꼬리를 가지며
중앙값을 포함한 자료가 오른쪽에 더 많이 분포해 있다.
평균과 중앙값이 같으면 왜도는 0이 된다.

첨도
확률분포의 꼬리가 두꺼운 정도를 나타내는 척도이다.
극단적인 편차 또는 이상치가 많을 수록 큰 값을 나타낸다.
첨도값(K)이 3에 가까우면 산포도가 정규분포에 가깝다.
3보다 작을 경우에는(K<3) 산포는 정규분포보다
꼬리가 얇은 분포로 생각할 수 있다.
"""
# 라이브러리 및 데이터 불러오기
# 'SalePrice'컬럼 왜도와 첨도계산
# 'SalePrice'컬럼 로그변환
# 'SalePrice'컬럼 왜도와 첨도계산
# 모두 더한 다음 출력
# %%
import pandas as pd
import numpy as np

df = pd.read_csv('archive/House Prices/train.csv')
# %%
df['SalePrice'].head()
# %% 왜도(skew), 첨도(kurt) 계산
s1 = df['SalePrice'].skew()
k1 = df['SalePrice'].kurt()
print("왜도: ", s1, "\n첨도: ", k1)
# %% 로그변환
df['SalePrice'] = np.log1p(df['SalePrice'])
# %% 왜도, 첨도 계산
s2 = df['SalePrice'].skew()
k2 = df['SalePrice'].kurt()
print("왜도: ", s2, "\n첨도: ", k2)
# %% 모두 더한 다음 출력
print(round(s1 + s2 + k1 + k2, 2))
