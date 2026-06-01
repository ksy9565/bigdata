# %%
"""
1. 30대 프리미엄 고객의 카테고리별 평균 구매금액 중
   ‘패션’ 카테고리가 차지하는 비율(%)은 얼마인가요?
   정답은 소수 첫째자리에서 반올림하여 정수로 입력하세요 (예: 38)
2. 프리미엄 고객의 ‘생활용품’ 평균 구매금액은 일반 고객보다 몇 원 더 높나요?
   정답은 정수로 입력하세요 (소수점 절사)
"""
import pandas as pd

df = pd.read_csv("archive/purchase.csv")
df_head = df.head()
# %%
# 소문제 1
# 1. 30대, 프리미엄
cond = (df["세그먼트"] == "프리미엄") & (df["연령대"] == "30대")
df_filtered = df[cond]

# 2. 카테고리별 평균 구매 금액
mean_purchase = df_filtered.groupby("카테고리")["구매금액"].mean()

# 3. 전체 평균의 합과 '패션' 카테고리의 비율 계산
fashion = round(mean_purchase["패션"] / mean_purchase.sum() * 100)
print(fashion)

# 소문제 2
# 1. 생활용품
cond = df["카테고리"] == "생활용품"
df_filtered = df[cond]

# 2. 고객 세그먼트별 평균 구매금액
mean_purchase = df_filtered.groupby("세그먼트")["구매금액"].mean()

# 3. 차이 계산
diff = int(mean_purchase["프리미엄"] - mean_purchase["일반"])
print(diff)
