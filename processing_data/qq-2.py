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


#flights 결측치만 제거한거
# 시간대별 운항량 집계
top3_carriers = ['B6', 'UA', 'EV']
top3_flights = flights_cleaned[flights_cleaned['carrier'].isin(top3_carriers)]
hourly_distribution = top3_flights.groupby(['carrier', 'hour']).size().unstack(fill_value=0)
print(hourly_distribution)

top3_carriers_cleaned = (
    flights_cleaned['carrier'].value_counts()
    .head(3)
    .index
    .tolist()
)
print("flights_cleaned 기준 상위 3개 항공사:", top3_carriers_cleaned)

# 테일넘+flights결측치 제거
merged_df = pd.merge(flights_cleaned, df_planes[['tailnum', 'seats']], on='tailnum', how='inner')
top3_carriers = ['B6', 'UA', 'EV']
top3_df = merged_df[merged_df['carrier'].isin(top3_carriers)]


# 시간대별 항공편 수 계산
flight_count_table = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='flight', 
    aggfunc='count'
).fillna(0).astype(int)
print(flight_count_table)
#6~17시에 모든 항공사 운항 집중
#새벽(0~4시) 없음
#B6 6-9,14,17,20 
# UA는 6-8,15,17사이가 피크
#EV는 6시, 8시, 14,16시,19시가 강세
# 3. 시간대별 총 좌석 수 계산 (pivot table)

flight_count_table.describe()

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

mean_seats_table = top3_df.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='mean'
).fillna(0).round(1)

#B6 
#좌석 수 총 기준으로는 6-7시,9시,16시,20시
#평균 좌석 수는 시간대별 변동 폭이 큼
#해석: 시간대에 따라 소형중형대형 혼합 

#EV
#총 좌석 수는 6시, 8시, 13~16시,19시 
#평균 좌석 수는 타항공사대비 적음
# 소형기 .
#23시에는 운항 없음.

#UA 
#좌석 수 총합도 크고, 운항편도 많음 
#대형기
#모든 시간대에 고르게 분포되어 있음


# 15분 이상 지연 항공편만 필터링
top3_carriers = ['B6', 'UA', 'EV']
delayed_top3 = flights_cleaned[
    (flights_cleaned['dep_delay'] >= 15) &
    (flights_cleaned['carrier'].isin(top3_carriers))
]

delay_hourly_counts = delayed_top3.pivot_table(
    index='hour',
    columns='carrier',
    values='flight',  # 항공편 번호 기준으로 count
    aggfunc='count',
    fill_value=0
)

print(delay_hourly_counts)
delay_hourly_counts.describe()
#b6는 14,16,17,20,21 지연 발생률 높음
#ev는 14-16,19-20
#ua는 15,17-20


#15분이상지연 테일넘 병합
merged_df = pd.merge(flights_cleaned, df_planes, on='tailnum', how='inner')
top3_carriers = ['B6', 'UA', 'EV']
top3_merged = merged_df[
    (merged_df['carrier'].isin(top3_carriers)) & 
    (merged_df['dep_delay'] >= 15) 
]

delay_flight_count = top3_merged.pivot_table(
    index='hour',
    columns='carrier',
    values='flight',
    aggfunc='count',
    fill_value=0
)
print(delay_flight_count)
delay_flight_count.describe()
avg_seats = top3_merged.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='mean',
    fill_value=0
).round(1)

sum_seats = top3_merged.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='sum',
    fill_value=0
).round(1)

max_seats = top3_merged.pivot_table(
    index='hour',
    columns='carrier',
    values='seats',
    aggfunc='max',
    fill_value=0
).round(1)

#B6 
#평균 좌석 수는 전체적으로 100~200석 범위로 
# 소형,중형기 운용
#총 좌석 수가 13~21시
#B6는 지연 항공편에서도 탄력적으로 기체를 
# 운영하며, 피크 시간대에 대형기를 집중 투입

# EV
#평균 좌석 수는 항상 50~60석
# 소형기
#총 좌석 수가 8~16시 사이
#EV는 지연 여부와 무관하게 하루 일정 시간대

#UA 
#평균 좌석 수보면 대형기 사용
#대부분 시간대에서 B6보다 많거나 비슷 
#UA는 지연이 발생해도 대형기 고정 운용

df_flights.info()
df_planes.info()

import matplotlib.pyplot as plt
top3_carriers = ['B6', 'UA', 'EV']
top3_flights = flights_cleaned[flights_cleaned['carrier'].isin(top3_carriers)]

# 전체 운항편 수
total_counts = top3_flights.groupby(['carrier', 'hour']).size().reset_index(name='total_count')

# 15분 이상 지연된 항공편 수
delayed_counts = top3_flights[top3_flights['dep_delay'] >= 15].groupby(['carrier', 'hour']).size().reset_index(name='delayed_count')

# 병합
merged_counts = pd.merge(total_counts, delayed_counts, on=['carrier', 'hour'], how='left').fillna(0)

# 시각화
for carrier in top3_carriers:
    data = merged_counts[merged_counts['carrier'] == carrier]

    plt.figure(figsize=(10, 5))
    plt.plot(data['hour'], data['total_count'], label='전체 운항편', linestyle='--', marker='o', color='green')
    plt.plot(data['hour'], data['delayed_count'], label='15분 이상 지연편', linestyle='-', marker='o', color='gold')

    plt.title(f'{carrier} 항공사 시간대별 운항편 수')
    plt.xlabel('시간대 (Hour)')
    plt.ylabel('항공편 수')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    