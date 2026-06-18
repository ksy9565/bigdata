맞습니다! 수험생들 사이에서 "이 템플릿 하나만 외워가면 무조건 합격 점수(혹은 만점) 받는다"라고 불리는 **국룰 템플릿**이 존재합니다.

빅분기 시험은 채점관이 코드를 하나하나 정성스럽게 읽는 것이 아니라, **제출된 CSV 파일의 예측 값과 실제 정답을 비교해 평가지표 점수(ROC-AUC 등)로 자동 채점**하기 때문입니다. 즉, 복잡한 기법 없이 기본 모델로 에러 없이 제출만 해도 만점이나 고득점이 나옵니다.

시험장에서 기계처럼 타이핑하고 나올 수 있는 **최신 기준(최근 기출 경향 반영) 작업형 2 국룰 템플릿**을 짜드릴 테니, 통째로 외우세요!

---

## 🏆 빅분기 작업형 2 국룰 템플릿 (암기용)

시험장 환경에서 아래 순서대로 구역을 나눠서 작성하면 절대 꼬이지 않습니다.

### 1단계: 데이터 불러오기 & 탐색 (EDA)

```python
import pandas as pd

# 1. 데이터 로드 (시험장에 주어지는 경로 입력)
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

# 2. 구조 확인 (시험장 화면 우측에서 log 찍어보며 눈으로 확인)
# print(X_train.info())
# print(X_train.isnull().sum()) # 결측치 확인

```

### 2단계: 데이터 전처리 (결측치, ID 제거, 인코딩)

```python
# 1. 수험번호/ID 열은 예측에 방해되므로 미리 저장하고 본문에서 삭제
X_test_id = X_test['ID'] # 나중에 제출 파일 만들 때 필요
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

# 2. 결측치 처리 (가장 무난한 중앙값이나 최빈값 채우기)
# 수치형 데이터는 중앙값, 범주형 데이터는 최빈값으로 채우는 게 안전합니다.
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_train.median(numeric_only=True)) # fit 기준 유지를 위해 train 중앙값 활용

# 3. 범주형 데이터 원-핫 인코딩 (국룰 get_dummies)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# ★ 중요: train과 test의 원-핫 인코딩 후 열 개수와 순서 맞추기
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

```

### 3단계: 데이터 분할 (검증 데이터셋 만들기)

```python
from sklearn.model_selection import train_test_split

# 모델이 잘 맞춰지는지 시험장에서 '셀프 채점'하기 위해 8:2로 나눕니다.
# y_train의 구조에 따라 컬럼명을 맞춰줍니다. (예: 'target')
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train['target'], test_size=0.2, random_state=42)

```

### 4단계: 모델링 & 셀프 검증 (국룰 모델: RandomForest)

```python
# [주의] 문제의 평가지표를 보고 Classifier와 Regressor 중 선택하세요!
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score # 시험지 하단 평가지표 수입

# 1. 모델 선언 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)

# 2. 검증 데이터로 셀프 채점해보기 (분류 예측 확률값 기준)
pred_val = model.predict_proba(X_val)[:, 1]
print("인공지능 자체 점수(ROC-AUC):", roc_auc_score(y_val, pred_val))

```

### 5단계: 최종 예측 및 파일 제출 (제일 중요!)

```python
# 1. 전체 train 데이터로 모델 재학습 (선택사항이나 성능 향상에 도움)
model.fit(X_train, y_train['target'])

# 2. 진짜 정답 파일로 만들 X_test 데이터 예측하기
pred_test = model.predict_proba(X_test)[:, 1]

# 3. 데이터프레임 만들고 csv로 내보내기 (문제에서 요구한 컬럼명 확인 필수)
submit = pd.DataFrame({
    'ID': X_test_id,
    'target': pred_test
})

submit.to_csv("result.csv", index=False)

# 4. 제대로 만들어졌는지 마지막 확인 (제출 전 필수 로그)
# print(pd.read_csv("result.csv").head())

```

---

## 🚨 국룰 템플릿 쓸 때 '이것' 안 하면 0점 나옵니다!

1. **`X_train`과 `X_test` 열 맞추기 (`align`)**
* `pd.get_dummies`를 그냥 쓰면 train에만 있는 글자(예: '포도') 때문에 train과 test의 열 개수가 달라져서 모델이 에러를 뿜어냅니다. 위 템플릿에 적어둔 `X_train.align(X_test, ...)` 코드가 그 에러를 원천 차단해 주는 마법의 코드입니다.


2. **`predict` 인지 `predict_proba` 인지 구별하기**
* 시험 문제가 **ROC-AUC** 점수를 원한다면 무조건 `predict_proba`를 써서 확률값을 제출해야지, `predict`로 0, 1 딱 떨어지게 내면 점수가 처참하게 깎입니다.


3. **`index=False` 꼭 넣기**
* `to_csv("result.csv")` 할 때 `index=False` 안 넣으면 맨 앞에 쓸데없는 순서 번호(0, 1, 2...)가 열로 추가되어 채점 서버에서 인식 오류로 0점 처리될 수 있습니다.



이 템플릿 구조를 빈 종이에 안 보고 3번 정도 타이핑할 수 있을 정도로 외워가시면, 시험장 변수가 생겨도 30분 안에 작업형 2번을 끝내고 웃으면서 나오실 수 있습니다. 혹시 이 코드 중에서 한 줄이라도 왜 쓰는지 이해가 안 가는 부분이 있다면 바로 말씀해 주세요!