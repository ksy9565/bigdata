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
# 라이브러리 불러오기
import pandas as pd

pd.set_option('display.max_columns', None)
# 데이터 불러오기
train = pd.read_csv("archive/4th/train.csv")
test = pd.read_csv("archive/4th/test.csv")

# EDA
print(train.info())
print(test.info())
print(train.head())
target = train.pop('Segmentation')

print(train.isnull().sum())
print(test.isnull().sum())

cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']

df = pd.concat([train, test], ignore_index=True)

from sklearn.preprocessing import LabelEncoder

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X_train = df.iloc[:len(train)].reset_index(drop=True)
test = df.iloc[len(train):].reset_index(drop=True)
print(X_train.info())

X_train = X_train.drop('ID', axis=1)
test_ID = test.pop('ID')

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=200, min_samples_split=5)
model.fit(X_tr, y_tr)
pred = model.predict(X_val)

from sklearn.metrics import f1_score

print("Macro f1 score: ", f1_score(y_val, pred, average='macro'))

pred = model.predict(test)

submit = pd.DataFrame({'ID': test_ID, 'Segmentation': pred})
print(submit)
