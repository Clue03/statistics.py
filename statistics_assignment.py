# 사용된 자료: 가해자_법규위반별_주야별_교통사고
# 자료의 정리 방법 3가지 이상 (1) 도표 (2) bar 그래프 (3) Line 그래프
# 대푯값 3개 (평균, 최빈값, 중앙값)
# 산포도 중 3개 이상
# 도수분포표 (k값은 적정하게 선택)

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
day_night = { x:y for x,y in zip(day,night) }

#-------자료의 정리---------------------------------------------------------------------------

# # 자료의 정리(1): 도표
# Df = pd.DataFrame(day_night, index=rows, columns=columns)
# # Df_t = Df.T
# # print(Df_t)
# print(Df)

#폰트 관련 코드
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = (12,8)

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

#자료의 정리(3): Line 그래프
plt.title('2021년 법규위반 주야별 교통사고')
plt.ylabel('교통사고 수')
plt.xlabel('법규위반별')

plt.plot(rows, day, linestyle='solid', c='r',label='주')
plt.plot(rows, night, linestyle='solid', c='b',label='야')
plt.legend(loc='upper right', ncol=1, fontsize=11)

#값 표시
for i, v in enumerate(rows):
    plt.text(v, night[i], night[i],
             fontsize = 7,
             fontweight = 'bold',
             color = 'black',
             horizontalalignment='center',
             verticalalignment='bottom')

for i, v in enumerate(rows):
    plt.text(v, day[i], day[i],
             fontsize = 7,
             fontweight = 'bold',
             color = 'black',
             horizontalalignment='center',
             verticalalignment='bottom')

plt.show()

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

# 산포도(1) : 왜도

# 산포도(2) : 첨도

# 산포도(3) : 상자 그림
