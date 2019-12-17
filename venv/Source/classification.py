from classification_util import ClassificationUtil

gildong = ClassificationUtil()

gildong.ignore_warning()

gildong.read("../Data/sleepdata.csv") # 파일 읽기
gildong.show() # 데이터 형태 보기

gildong.myplot("sleep_time", "intensity", "sleep") # sleep을 범례로 사용한 그래프
gildong.pairplots(["sleep_time", "intensity", "sleep"], "intensity") # 색을 구분하는 플롯 - 예외로 인해 색은 제외
gildong.lmplot("sleep_time", "intensity", "sleep") # 기본 plot과 차이가 없어보임
gildong.myhist() # 각 컬럼에 따른 히스토그램 그리기
gildong.myviolinplot("sleep", "intensity") # intensity에 따른 바오올린플롯 그리기
gildong.heatmap() # 각 컬럼이 서로 얼마나 관계가 깊은지 확인하는 히트맵 그리고
gildong.boxplot("sleep", "intensity") # intensity에 따른 sleep 상태 보기, 바이올린플롯과 모양이 다름
gildong.plot_3d("sleep_time", "intensity", "sleep") # 컬럼의 데이터 분포를 3d로 시각화

# 서프트벡터머신(Support Vector Machine), 논리회귀(Logistic Regression), KNN(K Nearest Neighbors),
#   의사결정트리(Desicion Tree)의 4가지 머신러닝 알고리즘을 이용한 분류 
gildong.run(["date", "time", "sleep_time"], ["sleep"], 1)