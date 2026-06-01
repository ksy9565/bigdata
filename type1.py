# %% 문제 1
"""
제공된 데이터(subject_performance.csv)는
학생들이 각 문제를 풀었을 때의 정답 여부를 기록한 자료이다.
각 문제는 대주제와 소주제로 분류되어 있다.
아래 수행 순서에 따라 데이터를 처리한 뒤,
구한 값을 [제출 형식]에 맞춰 답안 페이지에 입력하시오.

제공 데이터
- major_topic : 대주제
- sub_topic : 소주제
- student_id : 학생 ID
- is_correct : 정답 여부(1: 정답, 0: 오답)

수행 순서
1. 소주제별 정답률을 계산한다.
   정답률 = (해당 소주제의 정답 개수) / (해당 소주제의 전체 문제 풀이 수)

2. 소주제들의 정답률 중 3번째로 높은 값을 구하시오.
   단, 정답률이 같은 경우 하나로 취급한다.
   정답률이 0.7, 0.7, 0.6, 0.5인 경우, 세 번째로 높은 값은 0.5이다.

[제출형식] 반올림하여 소수 셋째 자리까지 작성
"""
import pandas as pd

df = pd.read_csv("archive/subject_performance.csv")
df_head = df.head()
# %% 방법1
# 1. 소주제별 정답률 계산
acc = df.groupby("sub_topic")["is_correct"].mean()

# 2. 내림차순 정렬 후 중복 제거
result = acc.sort_values(ascending=False).drop_duplicates()
print(round(result, 3))
# %% 방법2
# 1. 소주제별 정답률 계산
acc = df.groupby("sub_topic")["is_correct"].mean()

# 2. 상위 4개 추출 후 중복 제거
result = acc.nlargest(4).drop_duplicates()

print(round(result, 3))
# %% 3번째로 높은 정답률 출력
print(round(result.iloc[2], 3))

# %% 문제 2
"""
제공된 데이터(cafe_sales.csv)는 2024년 4월부터 2025년 12월까지의 카페 메뉴 판매 이력이다.
소문항을 각각 구한 다음 [제출 형식]에 맞춰 답안 페이지에 입력하시오.

제공 데이터
- order_date : 주문 날짜 (예 2024.05.24 13:22)
- category : 음료/디저트 종류 (예 coffee, tea …)
- item : 메뉴명
- price : 판매 금액

소문항
2-1. 연-월별 총 매출액을 계산하여 큰 순서대로 정렬했을 때, 2번째로 큰 매출액을 입력하시오.

2-2. 연-월별 총 매출액이 4번째로 큰 연-월을 찾으시오.
     해당 연-월의 데이터에서 카테고리별 매출 합계를 계산하고, 그중 최댓값을 입력하시오.

[제출 형식] 정수로 작성
"""
import pandas as pd

df = pd.read_csv("archive/cafe_sales.csv")
df_head = df.head()
# %% 2-1
# order_date -> 연-월 변환
df["order_date"] = pd.to_datetime(df["order_date"])
df["year_month"] = df["order_date"].dt.to_period("M")
df_head = df.head()

# 1. 연-월 단위로 그룹화하여 각 월의 총 매출액을 계산
monthly_sales = df.groupby("year_month")["price"].sum()

# 2. 연-월별 총 매출액 중 상위 2개만 추출
monthly_sales.nlargest(3)
# %% 2-2
# 1. 연-월별 총 매출액 중 상위 4개 추출
top4 = monthly_sales.nlargest(4)
target_ym = top4.index[3]

# 2. 4번째로 큰 연-월에 해당하는 데이터 추출
cond = df["year_month"] == target_ym
df = df[cond]

# 3. 해당 연-월의 카테고리별 매출 합계 중 최댓값 계산
cate = df.groupby("category")["price"].sum()

print(cate.max())

# %% 문제 3
"""
제공된 데이터(hamspam.csv)는 문자 메시지 내용과 스팸 여부를 담고 있다.
아래 수행 순서에 따라 데이터를 처리한 뒤,
구한 값을 [제출 형식]에 맞춰 답안 페이지에 입력하시오.

제공 데이터
- label : 메시지 구분 (spam: 스팸, ham: 정상)
- text : 메시지

수행 순서
1. 각 메시지의 단어 개수를 띄어쓰기(공백) 기준으로 계산하시오.
2. 스팸 메시지의 평균 단어 개수와 정상 메시지의 평균 단어 개수를 각각 구하시오.
3. 두 평균의 차이의 절댓값을 구하여 입력하시오.

[제출형식] 반올림하여 소수 셋째 자리까지 작성
"""
import pandas as pd

df = pd.read_csv("archive/hamspam.csv")
df_head = df.head()
# %% 방법2
# 1. 각 메시지의 단어 개수 구하기
df["단어"] = df["text"].str.split()
df["단어수"] = df["단어"].str.len()
df_head = df.head()

# 2. 스팸과 정상 메시지의 평균 단어 수 구하기
m = df.groupby("label")["단어수"].mean()

# 3. 두 평균의 차이의 절댓값 구하기
diff = abs(m["spam"] - m["ham"])

# 4. 소수 셋째 자리까지 반올림
print(round(diff, 3))

# %% 방법2
# 1. 각 메시지의 단어 개수 구하기
df["단어수"] = df["text"].str.count(" ") + 1
df_head = df.head()

# 2. 스팸과 정상 메시지의 평균 단어 수 구하기
cond1 = df["label"] == "spam"
cond2 = df["label"] == "ham"
m1 = df[cond1]["단어수"].mean()
m2 = df[cond2]["단어수"].mean()

# 3. 두 평균 차이의 절댓값 구하기
diff = abs(m1 - m2)

# 4. 소수 셋째 자리까지 반올림
print(round(diff, 3))
