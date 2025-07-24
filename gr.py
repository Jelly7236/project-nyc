import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nycflights13 as flights

df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather
##################################################################################################
## 데이터 전처리

# 결측치 제거
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned.shape # (327346, 19)

# flights_cleanded와 df_planes merge ( key = tailnum )
flights_cleaned = pd.merge(flights_cleaned,df_planes,on='tailnum',how='left')

# 시간순으로 날짜 재정렬
flights_cleaned.sort_values(['month', 'day'], inplace=True)
flights_cleaned = flights_cleaned.reset_index(drop=True) # 인덱스 초기화

# 15분 이상 지연된 항공편만 따로 데이터 프레임 생성 
flights_delay = flights_cleaned[flights_cleaned['dep_delay']>=15]
flights_delay = flights_delay.reset_index(drop=True)
flights_delay

flights_cleaned['month_day_time'] = pd.to_datetime({
    'year': df_flights['year'],
    'month': df_flights['month'],
    'day': df_flights['day'],
    'hour': df_flights['hour'],
    'minute': df_flights['minute']
})

EV_total = flights_cleaned[flights_cleaned['carrier']=='EV']
EV_flight=EV_total[['carrier','distance','air_time']]
#ev 항공사 기체 촤석 분포
# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# EV 항공사 좌석 정보 준비
EV_seats = EV_total[['seats']].dropna()

# 좌석 구간 설정
bins = [0, 50, 70, 90, 110, 130, 150, float('inf')]
labels = ['~50석', '51~70석', '71~90석', '91~110석', '111~130석', '131~150석', '151석 이상']
EV_seats['좌석구간'] = pd.cut(EV_seats['seats'], bins=bins, labels=labels, right=False)

# 좌석 구간별 비율 계산
seat_distribution = (
    EV_seats['좌석구간']
    .value_counts(normalize=True)
    .sort_index() * 100
)

# 시각화
plt.figure(figsize=(10, 6))
colors = sns.color_palette('pastel', len(seat_distribution))
barplot = sns.barplot(x=seat_distribution.index, y=seat_distribution.values, palette=colors)

# y축 범위 명시적으로 지정 (최대 100)
plt.ylim(0, 100)

# 제목 및 축 설정
plt.title('EV 항공사 기체 좌석 수 분포', fontsize=16, fontweight='bold')
plt.xlabel('좌석 수 구간', fontsize=12)
plt.ylabel('비율 (%)', fontsize=12)

# 수치 라벨 (막대 위에 충분히 띄워서)
for i, value in enumerate(seat_distribution.values):
    plt.text(i, value + 2, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()