import numpy as np
import pandas as pd

import pandas as pd
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])

df_flights = flights.flights
df_airlines = flights.airlines

flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned

##flight결측치만 제거한 상태에서 운행량?
# 항공사별 총 항공편 수 계산
flight_counts = flights_cleaned['carrier'].value_counts().reset_index()
flight_counts.columns = ['carrier', 'num_flights']

# 항공사 이름 붙이기
flight_counts = pd.merge(flight_counts, df_airlines, on='carrier', how='left')

top3_carriers = flights_cleaned['carrier'].value_counts().head(3).index.tolist()
print("Top 3 carriers:", top3_carriers)

# 시간대별 항공편 수 계산
hourly_distribution = flights_cleaned.groupby(['carrier', 'hour']).size().reset_index(name='count')

# 상위 3개 항공사만 필터링
filtered_distribution = hourly_distribution[hourly_distribution['carrier'].isin(top3_carriers)]
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_distribution, x='hour', y='count', hue='carrier', marker='o')
plt.title('Top 3 Airlines Flight Count by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Flights')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title='Carrier')
plt.tight_layout()
plt.show()

# 상위 3개 항공사 필터링
top3_df = flights_cleaned[flights_cleaned['carrier'].isin(top3_carriers)]

# 시간대별 운항 수 pivot 테이블 생성
pivot_table = top3_df.pivot_table(index='hour', columns='carrier', values='flight', aggfunc='count', fill_value=0)

# 정렬된 표 보기
pivot_table = pivot_table.sort_index()
print(pivot_table)

# UA 항공사의 1시 출발 항공편만 필터링
ua_1am = flights_cleaned[(flights_cleaned['carrier'] == 'UA') & (flights_cleaned['hour'] == 1)]

# 결과 확인
print(ua_1am)
print(f"총 편수: {ua_1am.shape[0]}")

##flight,tailnum결측치만 제거한 상태에서 운행량?
merged_df = pd.merge(df_flights, df_planes, on='tailnum', how='inner')

top3_carriers = merged_df['carrier'].value_counts().head(3).index.tolist()
print("상위 3개 항공사:", top3_carriers)
top3_df = merged_df[merged_df['carrier'].isin(top3_carriers)]

hourly_distribution = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='flight',
    aggfunc='count',
    fill_value=0
).sort_index()

print(hourly_distribution)

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_distribution)
plt.title('Top 3 항공사 시간대별 운항편 수')
plt.xlabel('시간대 (Hour)')
plt.ylabel('운항편 수')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title='항공사')
plt.tight_layout()
plt.show()

hourly_distribution.plot(kind='bar', figsize=(14, 6))
plt.title('Top 3 항공사 시간대별 운항편 수 (막대그래프)')
plt.xlabel('시간대 (Hour)')
plt.ylabel('운항편 수')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(title='항공사')
plt.tight_layout()
plt.show()

# tailnum,flights결측치 시간대별 평균 좌석 수
avg_seats_by_hour = merged_df.groupby('hour')['seats'].mean()

# 시각화
import matplotlib.pyplot as plt

top3_carriers = merged_df['carrier'].value_counts().head(3).index.tolist()
top3_df = merged_df[merged_df['carrier'].isin(top3_carriers)]

# 시간대별 평균 좌석 수 계산
avg_seats_table = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='mean'
).fillna(0).round(1)  # 소수 첫째자리까지 반올림

top3_carriers = merged_df['carrier'].value_counts().head(3).index.tolist()
top3_df = merged_df[merged_df['carrier'].isin(top3_carriers)]

# 시간대별 총 좌석 수 계산
sum_seats_table = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='sum'
).fillna(0).round(1)

max_seats_table = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='max'
).fillna(0).round(1)

#15분이상 출발지연 플라이트 결측치만
# 항공사별 시간대별 15분 이상 지연 항공편 수
top3_carriers = ['B6', 'UA', 'EV']
delayed_top3 = flights_cleaned[
    (flights_cleaned['carrier'].isin(top3_carriers)) &
    (flights_cleaned['dep_delay'] >= 15)
]

# 집계
hourly_delay_counts = delayed_top3.groupby(['carrier', 'hour']).size().reset_index(name='delayed_flight_count')

# 피벗 테이블로 보기 좋게
pivot_delay = hourly_delay_counts.pivot(index='hour', columns='carrier', values='delayed_flight_count').fillna(0)

# 결과 출력
print(pivot_delay)

# flights_cleaned 데이터에서 tailnum 기준으로 df_planes와 병합
merged_df = pd.merge(flights_cleaned, df_planes, on='tailnum', how='inner')

# 15분 이상 지연된 항공편만 추출
delayed_merged_df = merged_df[merged_df['dep_delay'] >= 15]
# 상위 3개 항공사 필터링
top3_carriers = ['UA', 'B6', 'EV']
top3_delayed = delayed_merged_df[delayed_merged_df['carrier'].isin(top3_carriers)]

# 시간대별 항공편 수 계산 (지연 15분 이상)
hourly_distribution = top3_delayed.groupby(['carrier', 'hour']).size().unstack(fill_value=0)

# 확인
print(hourly_distribution)

flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned 