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

# 결측치 제거
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])

##월별로 고위험 날씨 조건이 발생한 비율

# 월 컬럼 생성
df_weather['time_hour'] = pd.to_datetime(df_weather['time_hour'], errors='coerce')
df_weather['month'] = df_weather['time_hour'].dt.month

# 고위험 조건 
df_weather['high_risk'] = (
    (df_weather['wind_gust'] >= 30) |
    (df_weather['wind_speed'] >= 20) |
    (df_weather['visib'] < 1)
)

# 결측치 제거
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib'])

# 공항 × 월별 고위험 비율 계산
risk_by_month_airport = (
    df_weather_clean.groupby(['origin', 'month'])
    .agg(
        total_obs=('high_risk', 'count'),
        high_risk_obs=('high_risk', 'sum')
    )
    .reset_index()
)

#비율 계산
risk_by_month_airport['risk_ratio'] = (risk_by_month_airport['high_risk_obs'] / risk_by_month_airport['total_obs'] * 100).round(2)

#출력
print(risk_by_month_airport)


##뉴욕 3개 공항에서 1~6월 동안 항공사별 월간운항 횟수
# 1~6월 & 주요 항공사 & 주요 공항만 필터링
top3_carriers = ['UA', 'B6', 'EV']
filtered = flights_cleaned[
    (flights_cleaned['carrier'].isin(top3_carriers)) &
    (flights_cleaned['month'].between(1, 6)) &
    (flights_cleaned['origin'].isin(['EWR', 'JFK', 'LGA']))
]

# 항공사별 × 공항별 × 월별 운항 수 집계
monthly_airport_counts = (
    filtered.groupby(['carrier', 'month', 'origin'])
    .size()
    .reset_index(name='flight_count')
)

# 피벗 테이블로 보기 좋게 정리
pivot_table = monthly_airport_counts.pivot_table(
    index=['carrier', 'month'],
    columns='origin',
    values='flight_count',
    fill_value=0
).astype(int)

print(pivot_table)
pivot_table

##EV, UA, B6 각 항공사에서
##출발지연이 가장 심한 항공편 5개의 날씨

import pandas as pd
import numpy as np
import nycflights13 as flights

# 1. 데이터 불러오기
df_flights = flights.flights
df_weather = flights.weather

# 2. 결측치 제거
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay','arr_time','arr_delay','tailnum','air_time']).copy()

# 3. 날짜 컬럼 생성
flights_cleaned['date'] = pd.to_datetime(flights_cleaned[['year', 'month', 'day']]).dt.date
df_weather['date'] = pd.to_datetime(df_weather['time_hour']).dt.date

# 4. 주요 항공사 필터링
target_carriers = ['EV', 'UA', 'B6']
flights_filtered = flights_cleaned[flights_cleaned['carrier'].isin(target_carriers)].copy()

# 5. 출발 지연 상위 5개씩 추출
worst_departure_delays = (
    flights_filtered.sort_values(['carrier', 'dep_delay'], ascending=[True, False])
    .groupby('carrier')
    .head(5)
    .copy()
)

# 6. 공항별+날짜별 평균 날씨 정보 생성
weather_daily_avg = df_weather.groupby(['origin', 'date'])[
    ['wind_gust', 'wind_speed', 'visib', 'temp', 'humid', 'precip']
].mean().reset_index()

# 7. 항공편과 날씨 병합
merged = pd.merge(
    worst_departure_delays,
    weather_daily_avg,
    on=['origin', 'date'],
    how='left'
)

# 8. 결과 정리
result = merged[['carrier', 'flight', 'origin', 'date', 'dep_delay', 'wind_gust', 'wind_speed', 'visib', 'temp', 'humid', 'precip']]
result.reset_index(drop=True, inplace=True)

# 9. 출력
print(result)
#지연심한 날짜에 날씨 영향없는 
# 결과로 나오는것만 찾을수 있는데
#일단 이정도만 보내고 더 찾아볼게요