"""
Q. [마케팅] 자동차 시장 세분화
자동차 회사는 새로운 전략을 수립하기 위해 4개의 시장으로 세분화했습니다.
기존 고객 분류 자료를 바탕으로 신규 고객이 어떤 분류에 속할지 예측해주세요!
- 예측할 값(y): "Segmentation" (1,2,3,4)
- 평가: Macro f1-score
- data: train.csv, test.csv
- 제출 형식:
    ID,Segmentation
    458989,1
    458994,2
    459000,3
    459003,4

답안 제출 참고
- 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
- pd.DataFrame({'ID': test.ID, 'Segmentation': pred})
    .to_csv('003000000.csv', index=False)

노트북 구분
- basic: 수치형 데이터만 활용 -> 학습 및 test데이터 예측
- intermediate: 범주형 데이터도 활용 -> 학습 및 test데이터 예측
- advanced: 학습 및 교차 검증(모델 평가) -> 하이퍼파라미터 튜닝 -> test데이터 예측

학습을 위한 채점
- 최종 파일을 "수험번호.csv"가 아닌 "submission.csv" 작성 후
  오른쪽 메뉴 아래 "submit" 버튼 클릭 -> 리더보드에 점수 및 등수 확인 가능함
- pd.DataFrame({'ID': test.ID, 'Segmentation': pred})
    .to_csv('submission.csv', index=False)
"""
# %%
# 라이브러리 불러오기
import pandas as pd

# 데이터 불러오기
train = pd.read_csv("archive/4th-t2/train.csv")
test = pd.read_csv("archive/4th-t2/test.csv")
# %% basic 단계
# EDA
train.shape, test.shape
# %%
# train 샘플 확인
train.head()
# %%
# target 확인
train['Segmentation'].value_counts()
# %%
# 결측치 확인
train.isnull().sum(), test.isnull().sum()
# %%
# type 확인
train.info()
# %%
train.describe()
# %% 전처리
target = train.pop('Segmentation')
target
# %% 수치형 데이터만 활용하여 학습(basic)
num_cols = ['Age', 'Work_Experience', 'Family_Size']
train = train[num_cols]
# %%
test['ID']
