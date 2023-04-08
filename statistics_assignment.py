# 사용된 자료: 가해자_법규위반별_주야별_교통사고, 청소년건강행태조사(2019년)
# 자료의 정리 방법 3가지 이상 (1) pie 그래프(주&야) (2) bar 그래프 (3) Line 그래프
# 대푯값 3개 (평균, 최빈값, 중앙값)
# 산포도 중 3개 이상 (왜도, 첨도, 상자 그림)
# 도수분포표 (k값은 적정하게 선택)

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import math
import csv

# csv 파일 불러오기
df = pd.read_csv('D:\Programming\Python\data1.csv', encoding='cp949')

row = list(df['가해자법규위반별'])
column = list(df['주야별'])

columns = [] # 주 & 야
rows = [] # 법규 위반별

for value in column:
    if value not in columns:
        columns.append(value)

for value in row:
    if value not in rows:
        rows.append(value)        
del rows[1]
rows.insert(11,'앞지르기방법위반')
print(rows)

values = list(df['2021 사고건수'])

day = [] # 주의 사고건수
night = [] # 야의 사고건수

for i in range(0, len(values), 1):
    if(i % 2 == 0):
        day.append(values[i])
    else:
        night.append(values[i])





#-------자료의 정리---------------------------------------------------------------------------

#폰트 관련 코드
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = (12,8)

# # 자료의 정리(1): Pie 그래프
# plt.pie(day, labels = day)
# plt.legend(rows, title='법규 위반', loc="center right",  bbox_to_anchor=(1, 0, 0.5, 1))
# plt.title("2021년 법규위반 교통사고(주)")
# plt.show()

# plt.pie(night, labels = night)
# plt.legend(rows, title='법규 위반', loc="center right",  bbox_to_anchor=(1, 0, 0.5, 1))
# plt.title("2021년 법규위반 주야별 교통사고(야)")
# plt.show()

# # 자료의 정리(2): Bar 그래프
# plt.figure()
# y = np.arange(len(day))  

# plt.barh(y - 0.2, day, height=0.4, color='#ff6600')
# plt.barh(y + 0.2, night, height=0.4, color='#0066ff')
# plt.yticks(y, rows)

# plt.legend()
# plt.title('2021년 법규위반 주야별 교통사고')
# plt.xlabel('교통사고 수')
# plt.ylabel('법규위반별')
# plt.xlim(0, 150)

# plt.show()

# #자료의 정리(3): Line 그래프
# plt.title('2021년 법규위반 주야별 교통사고')
# plt.ylabel('교통사고 수')
# plt.xlabel('법규위반별')

# plt.plot(rows, day, linestyle='solid', c='r',label='주')
# plt.plot(rows, night, linestyle='solid', c='b',label='야')
# plt.legend(loc='upper right', ncol=1, fontsize=11)

# #값 표시
# for i, v in enumerate(rows):
#     plt.text(v, night[i], night[i],
#              fontsize = 7,
#              fontweight = 'bold',
#              color = 'black',
#              horizontalalignment='center',
#              verticalalignment='bottom')

# for i, v in enumerate(rows):
#     plt.text(v, day[i], day[i],
#              fontsize = 7,
#              fontweight = 'bold',
#              color = 'black',
#              horizontalalignment='center',
#              verticalalignment='bottom')

# plt.show()

#-------대푯값----------------------------------------------------------------------

# # 평균
# print(f"연평균 사고건수: 약{sum(values) / 19:.1f}건")

# # 중앙값
# M = np.median(values)
# print('중앙값:',int(M))

# # 최빈값
# counts = Counter(values)
# counts.most_common()

# print(f'최빈값: {counts.most_common(3)}') # 최빈값 상위 3개 

#-------산포도----------------------------------------------------------------------------

# # 산포도(1) : 왜도
# def skew_kurtosis(day, night):
#     d_skew = stats.skew(day)    
#     n_skew = stats.skew(night)    
#     return d_skew, n_skew

# skew1, skew2 = skew_kurtosis(day, night)
# print("Skewness at day: ", skew1)
# print("Skewness at night: ", skew2)

# # 산포도(2) : 첨도
# def skew_kurtosis(day, night):
#     d_kurtosis = stats.kurtosis(day)   
#     n_kurtosis = stats.kurtosis(night)   
#     return d_kurtosis, n_kurtosis

# kurt1, kurt2 = skew_kurtosis(day, night)
# print("Kurtosis at day: ", kurt1)
# print("Kurtosis at night: ", kurt2)

# # 산포도(3) : 상자 그림
# plt.boxplot([day, night], vert=False, showmeans=True)
# plt.grid()
# plt.show()

#-------도수분포표----------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = (5, 3)        # (가로,세로) 인치 단위
plt.rcParams['axes.unicode_minus'] = False    # 그래프 눈금 값에서 (-)숫자표시
plt.rcParams['lines.linewidth'] = 2            # 선 굵기
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 사용  (AppleGothic)


def makeFrequencyTable(data, k = 6):
    R = round(max(data)-min(data), 6) # 2.R : 최대측정값 - 최소측정값
    w = math.ceil(R / k)          # 3.계급 간격
    s = min(data) - 0.5        # 4.시작 계급값
    bins = np.arange(s, max(data)+w+1, step=w)  #계급
    index = [f'{bins[i]} ~ {bins[i+1]}' for i in range(len(bins)) if i<(len(bins)-1) ] # 계급 구간(index)
    hist, bins = np.histogram(data, bins)  # 계급 구간별 도수 데이터
    print(f'계급수(K):{k}, R:{R}, 계급간격(w):{w}, 계급시작값(s):{s}')
    print(f'계급:{bins}')

    # 도수분포표 만들기
    DF = pd.DataFrame(hist, index=index, columns=['도수'])
    DF.index.name = '계급간격'

    DF['상대도수'] = [x/sum(hist) for x in hist]
    DF['누적도수'] = [sum(hist[:i+1]) if i>0 else hist[i] for i in range(k + 1)]
    DF['누적상대도수'] = [sum(hist[:i+1]) if i>0 else DF['상대도수'].values[i] for i in range(k + 1)] 
    DF['계급값'] = [ (bins[x]+bins[x+1])/2 for x in range(k + 1)]

    return DF

# 도수분포표 전용 파일 불러오기
data = pd.read_csv("D:\Programming\Python\Data_for_study.csv")
tall = list(data['키'])

dosu = makeFrequencyTable(tall, k = 6)
print(dosu)
