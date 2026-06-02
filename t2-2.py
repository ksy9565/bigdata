# %%
"""
당뇨병 여부 판단
예측컬럼: Outcome (0 정상, 1 당뇨) 당뇨병일 확률 예측
평가지표: roc-auc
제출파일명: result.csv (1개컬럼, 컬럼명 pred)
"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
train = pd.read_csv("archive/diabetes_train.csv")
test = pd.read_csv("archive/diabetes_test.csv")
print(train.shape, test.shape)
# %% EDA
train.head()
# %%
train['Outcome'].value_counts()
# %%
train.info()
# %%
train.isnull().sum()
# %%
test.isnull().sum()
# %%
train.describe()
# %% 전처리
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])
# %% id 제외
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
# %% 타겟
target = train.pop("Outcome")
# %% 학습 및 예측
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=2022)
model.fit(train, target)
predictions = model.predict_proba(test)
# %% 당뇨 확률값 선택
pred = predictions[:, 1]
# %% 파일 생성
output = pd.DataFrame({'pred': pred})
output.head()
