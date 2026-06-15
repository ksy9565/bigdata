"""
문제: 두 교육 방법의 효과 비교
연구자는 두 가지 다른 교육 방법이 학생들의 성적에 미치는 영향을 비교하고자 합니다.
연구자는 무작위로 선발된 20명의 학생들을 두 그룹으로 나누어
한 그룹에는 교육 방법 A를, 다른 그룹에는 교육 방법 B를 적용합니다.
교육이 끝난 후, 두 그룹의 성적을 비교하기 위해
독립 표본 t-검정과 ANOVA F-검정을 실시하려고 합니다.

다음은 두 그룹의 성적입니다:
다음의 두 가지 검정을 사용하여 두 교육 방법 간의 성적 차이가
통계적으로 유의한지를 검증하세요
  1. 독립 표본 t-검정을 실시하여 t 통계량을 구하세요.
  2. 독립 표본 t-검정을 실시하여 p-값을 구하세요.
  3. ANOVA F-검정을 실시하여 F 통계량을 구하세요.
  4. ANOVA F-검정을 실시하여 p-값을 구하세요.
"""
# %%
import pandas as pd

df = pd.DataFrame({
    'A': [77, 75, 82, 80, 81, 83, 84, 76, 75, 87],
    'B': [80, 74, 77, 79, 71, 74, 78, 69, 70, 72],
})
# %%
from scipy import stats

t_statistic, t_p_value = stats.ttest_ind(df['A'], df['B'])
print(f"t 통계량: {t_statistic:.5f}, p값: {t_p_value:.5f}")
# 다음으로 ANOVA F-검정을 수행합니다.
f_statistic, f_p_value = stats.f_oneway(df['A'], df['B'])
print(f"F 통계량: {f_statistic:.5f}, p값: {f_p_value:.5f}")
