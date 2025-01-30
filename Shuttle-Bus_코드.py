# 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
from scipy.linalg import eig
import squarify
from scipy.stats import ks_2samp, wilcoxon, ttest_ind

# 폰트 설정 (한국어 깨짐 방지)
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# 1. 막대그래프 4개 그리기
# 전국, 서울, 경기도 대학생 교통수단별 이용자 수 시각화
files = ["C:/Users/rlaek/OneDrive/바탕 화면/4-2/통자분/서론 데이터/전국_교통수단.xlsx", "C:/Users/rlaek/OneDrive/바탕 화면/4-2/통자분/서론 데이터/서울_교통수단2.xlsx",
         "C:/Users/rlaek/OneDrive/바탕 화면/4-2/통자분/서론 데이터/경기도_교통수단.xlsx"]
titles = ["전국", "서울", "경기도"]
for file, title in zip(files, titles):
    df = pd.read_excel(file)
    filtered_df = df[df['교육정도별'].isin(['대학(4년제 미만)', '대학교(4년제 이상)'])]
    summed_df = filtered_df.drop(columns=['교육정도별']).sum()
    plt.figure(figsize=(10, 6))
    summed_df.plot(kind='bar')
    plt.title(f'{title} 대학생 교통수단별 이용자 수', fontproperties=font_prop)
    plt.xlabel('교통수단', fontproperties=font_prop)
    plt.ylabel('이용자 수 합계', fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop, rotation=45, ha='right')
    plt.show()

# 지역별 통학 시간 시각화
df4 = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/통자분/데이터/2023 통학시간 데이터.xlsx")
selected_columns = ['소계', '거주시군내', '도내다른 시군', '서울', '인천', '기타']
summed_df4 = df4[selected_columns].sum()
plt.figure(figsize=(10, 6))
summed_df4.plot(kind='bar')
plt.title('지역별 통학 시간', fontproperties=font_prop)
plt.xlabel('지역', fontproperties=font_prop)
plt.ylabel('통학 시간 합계', fontproperties=font_prop)
plt.xticks(fontproperties=font_prop, rotation=45, ha='right')
plt.show()

# 2. 워드 클라우드 2개
survey = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/통자분/데이터/설문조사_퀴즈(응답).xlsx", sheet_name="설문지 응답 시트1")
survey.rename(columns={
    '2-8. 통학 환경에 가장 큰 영향을 주는 요소가 무엇인가요?': "통학 환경 영향 요소",
    '2-10. 경기대학교의 통학 문제 해결을 위해 가장 필요한 것은 무엇인가요?': "통학 문제 해결책"
}, inplace=True)

# 워드 클라우드 생성 (통학 환경 영향 요소)
text_data = survey['통학 환경 영향 요소'].dropna()
text = ' '.join(text_data)
wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("통학 환경 영향 요소", fontproperties=font_prop)
plt.show()

# 워드 클라우드 생성 (통학 문제 해결책)
text_data = survey['통학 문제 해결책'].dropna()
text = ' '.join(text_data)
wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("통학 문제 해결책", fontproperties=font_prop)
plt.show()

# 3. 가설 검정
거주지 = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/통자분/데이터/거주지.xlsx")
result_df = 거주지['거주지'].value_counts().reset_index().rename(columns={'index': '거주지', '거주지': '인원수'})
result_df.to_excel("거주지 인원수.xlsx", index=False)
df5 = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/4-2/통자분/modified6_data.xlsx")

# 남자26세_여자24세와 설문조사 count 칼럼을 정규화
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df5[['남자26세_여자24세', '설문조사count']])
df5['남자26세_여자24세_normalized'] = data_normalized[:, 0]
df5['설문조사count_normalized'] = data_normalized[:, 1]

# 정규화된 데이터의 분포 유사성 검정 (Kolmogorov-Smirnov 검정)
ks_stat, ks_p_value = ks_2samp(df5['남자26세_여자24세_normalized'], df5['설문조사count_normalized'])
print(f"Kolmogorov-Smirnov 검정 결과: 통계량={ks_stat}, p-값={ks_p_value}")

# 4. 상관관계 히트맵
final_df = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/통자분/데이터/최종데이터프레임(설문조사 지역 반영) (4).xlsx")
final_df['통행량 평균'] = (final_df['출근시간 통행량 합계'] + final_df['점심시간 통행량 합계'] + final_df['퇴근시간 통행량 합계']) / 3
data = final_df[['대중교통이용자인원(합계)', '정류장공급도', '통행량 평균', '20대거주인구', '가정기반 통학', '경기대까지의소요시간']]
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={'fontsize': 10}, cbar_kws={'shrink': 0.8})
plt.title("컬럼 간 상관관계 히트맵", fontproperties=font_prop)
plt.xticks(fontproperties=font_prop, rotation=45, ha='right')
plt.yticks(fontproperties=font_prop)
plt.tight_layout()
plt.show()

# 5. VIF 지수 계산
data_with_const = add_constant(data)
vif = pd.DataFrame()
vif["변수"] = data_with_const.columns
vif["VIF"] = [variance_inflation_factor(data_with_const.values, i) for i in range(data_with_const.shape[1])]
print(vif)

# 6. 고유벡터, 고유값 계산
data = final_df[['정류장공급도', '통행량 평균', '20대거주인구', '가정기반 통학', '경기대까지의소요시간']]
data_scaled = StandardScaler().fit_transform(data)
covariance_matrix = np.cov(data_scaled.T)
values, vectors = np.linalg.eig(covariance_matrix)
print('\nEigenvalues:\n', values)
print('\nEigenvectors:\n', vectors)

# PCA 분석 및 Scree Plot
pca = PCA(n_components=5)
pca.fit(data_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot', fontproperties=font_prop)
plt.xlabel('Principal Component', fontproperties=font_prop)
plt.ylabel('Explained Variance Ratio', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 첫 번째 주성분의 기여도 계산
eigenvector1 = pca.components_[0]
contributions = pd.Series(eigenvector1**2 / sum(eigenvector1**2) * 100)
print("첫 번째 주성분에서 각 변수의 기여도 (%):")
print(contributions)

# 7. AHP 분석: 쌍대비교 행렬을 통한 변수별 가중치
pairwise_matrix = np.array([
    [1,   2,   1/3, 1/5, 3],
    [1/2, 1,   1/5, 1/7, 1/3],
    [3,   5,   1,   1/2, 7],
    [5,   7,   2,   1,   8],
    [1/3, 3,   1/7, 1/8, 1]
])

# 일관성 비율 계산 함수
def consistency_ratio(matrix):
    eigvals, _ = eig(matrix)
    max_eigval = np.max(eigvals).real
    n = matrix.shape[0]
    CI = (max_eigval - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_dict.get(n, 1.45)
    CR = CI / RI if RI != 0 else 0
    return CR

CR = consistency_ratio(pairwise_matrix)
print("일관성 비율(CR):", CR)

# 가중치 계산 함수
def calculate_weights(matrix):
    eigvals, eigvecs = eig(matrix)
    max_eigval_index = np.argmax(eigvals)
    max_eigvec = eigvecs[:, max_eigval_index].real
    weights = max_eigvec / np.sum(max_eigvec)
    return weights

weights = calculate_weights(pairwise_matrix)
print("변수 가중치:", weights)

# 8. TOPSIS 분석
data_topsis = final_df[['정류장공급도', '가정기반 통학', '20대거주인구', '통행량 평균', '경기대까지의소요시간']]
criteria = [min, max, max, max, max]
weights = [0.29855775, 0.04806188, 0.11385963, 0.06614968, 0.47337105]

# 정규화
norm_data = data_topsis.values / np.sqrt((data_topsis.values ** 2).sum(axis=0))
weighted_data = norm_data * weights

# 이상해 및 비이상해 솔루션
ideal_best = [criterion(weighted_data[:, j]) for j, criterion in enumerate(criteria)]
ideal_worst = [criterion(weighted_data[:, j]) for j, criterion in enumerate([max if c == min else min for c in criteria])]

# 거리 계산
dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

# 상대적 근접성 점수 계산
scores = dist_worst / (dist_best + dist_worst)
final_df['TOPSIS_Score'] = scores
final_df['Rank'] = final_df['TOPSIS_Score'].rank(ascending=False)

# 9. 최종 결과 확인
sorted_df = final_df[['시군구', 'TOPSIS_Score', 'Rank']].sort_values(by='Rank')
print("TOPSIS 분석 결과:")
print(sorted_df)
