# %%
"""
2024년 1~3월 동안 부서별 초과근무시간을 분석한 결과,
특정 부서에서만 모든 달 평균 초과근무시간이 3시간을 초과한 것으로 나타났다.
이 부서는 인사팀에 보고되어 근무환경 개선 요청이 접수되었다.
부서의 이름은?
"""
import pandas as pd

df = pd.read_csv("archive/overwork.csv")
df_head = df.head()
# %%
df["날짜"] = pd.to_datetime(df["날짜"])
df["월"] = df["날짜"].dt.to_period("M")

# 피벗 테이블 생성
pivot = pd.pivot_table(df, index="부서", columns="월", values="초과근무시간", aggfunc="mean")
print(pivot)
# %%
result = df.groupby(["부서", "월"])["초과근무시간"].mean()
print(result)
# 마케팅
