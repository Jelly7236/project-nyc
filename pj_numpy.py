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


# 예시: 항공편 데이터 확인
print(df_flights.head())
df_flights
df_airports
df_planes['tailnum'](N10156)
df_weather


def count_daily_flights(df_flights, tailnum, year, month, day):
    result = df_flights[
        (df_flights['tailnum'] == tailnum) &
        (df_flights['year'] == year) &
        (df_flights['month'] == month) &
        (df_flights['day'] == day)
    ]
    return result.shape[0], result

# 예시 실행
count, records = count_daily_flights(df_flights, 'N535MQ', 2013, 1, 1)
print(f"2013-01-01에 N14228 운항 횟수: {count}")
df_weather['wind_speed'].describe()
max_idx = df_weather['wind_speed'].idxmax()
df_weather[df_weather['wind_speed'] == 1048.360580].head()

df_flights['arr_delay'].describe()
df_flights[df_flights['arr_delay'] == 1272.0000].head()
df_0212 = df_flights[
    (df_flights['time_hour'] == '2013-01-09T14:00:00Z') &
    (df_flights['origin'] == EWR) ]

df_flights[(df_flights['time_hour'] == '2013-01-09T14:00:00Z')& 
    (df_flights['origin'] == 'EWR')].head()

df_weather.describe()
jfk = df_weather[df_weather['origin'] == 'JFK']
lga = df_weather[df_weather['origin'] == 'LGA']
ewr = df_weather[df_weather['origin'] == 'EWR']
df_weather.shape[0]
jfk.describe() 
lga.describe()
ewr.describe()

#보통 wind_gust ≥ 30 mph 이상이면 항공사들이 지연대기
#더 민감한 항공기는 25mph만 넘어도 조치함

# 돌풍 조건 필터링
wind_30 = df_weather[df_weather['wind_gust'] >= 30]

# 공항(origin)별 돌풍 발생 횟수
gust_count = wind_30['origin'].value_counts().reset_index()
print(gust_count)
#jfk:404,lga:313,ewr:219

# 돌풍 조건 필터링
wind_25 = df_weather[df_weather['wind_gust'] >= 25]

# 공항(origin)별 돌풍 발생 횟수
gust_count = wind_25['origin'].value_counts().reset_index()
print(gust_count)
#jfk:969,lga:927,ewr:714

len(df_weather)

h_wind=df_weather[df_weather['wind_speed'] >= 20]

high_count = h_wind['origin'].value_counts().reset_index()
print(high_count)

#jfk:691,lga:475,ewr:300

#wind_gust - wind_speed ≥ 10
gust_diff = df_weather.dropna(subset=['wind_gust', 'wind_speed'])
gust_diff = gust_diff[(gust_diff['wind_gust'] - gust_diff['wind_speed']) >= 10]

gust_diff.shape[0]

gust_airport = gust_diff['origin'].value_counts().reset_index()

gust_airport

#lga:539,ewr:490,jfk:385

#visib < 1 조건을 만족하는 저시정(시정 불량)

low_visibility = df_weather.dropna(subset=['visib'])
low_visibility = low_visibility[low_visibility['visib'] < 1]

low_visibility.shape[0]

low_count = low_visibility['origin'].value_counts().reset_index()

#jfk:193,ewr:96,lga:90

#wind_gust ≥ 30, wind_speed ≥ 20, visib < 1하나라도

# 1. wind_gust, wind_speed, visib의 결측치를 제거하고 복사본 생성
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib']).copy()

# 2. 고위험 조건 플래그 컬럼 생성 (True/False)
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |      # 돌풍 강함
    (df_weather_clean['wind_speed'] >= 20) |     # 강풍
    (df_weather_clean['visib'] < 1)              # 시정 불량
)
# 3. 공항별 전체 관측 수 계산 (value_counts 결과는 Series)
total_by_origin = df_weather_clean['origin'].value_counts().reset_index()

#lga:2028,ewr:1802,jfk:1507

#월별 지연 발생 조건
# 1. 결측치 제거 및 복사본 생성
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib']).copy()

# 2. 고위험 조건 정의
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['visib'] < 1)
)

# 3. 전체 관측 수: 월별
total_by_month = df_weather_clean['month'].value_counts().sort_index().reset_index()
total_by_month.columns = ['month', 'total_obs']  # 명확히 컬럼명 지정

# 4. 고위험 조건 수: 월별
high_risk_by_month = df_weather_clean[df_weather_clean['high_risk']]['month'].value_counts().sort_index().reset_index()
high_risk_by_month.columns = ['month', 'high_risk_obs']

# 5. 병합
month_risk_ratio = pd.merge(total_by_month, high_risk_by_month, on='month', how='left').fillna(0)

# 6. 비율 계산
month_risk_ratio['risk_ratio'] = month_risk_ratio['high_risk_obs'] / month_risk_ratio['total_obs']

# 7. 결과 출력
print(month_risk_ratio)
month_risk_ratio['risk_ratio'].max() 
#2월에 최고 발생3월 2등,최소발생 7월 

#공항별 월별 
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib']).copy()

#고위험 조건 정의
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['visib'] < 1)
)

# 공항별 × 월별 고위험 조건 발생 횟수
high_risk_count = df_weather_clean[df_weather_clean['high_risk']].groupby(['origin', 'month']).size().reset_index(name='high_risk_count')

print(high_risk_count)

max_risk = high_risk_count.loc[high_risk_count.groupby('origin')['high_risk_count'].idxmax()].reset_index(drop=True)
#ewr:3월 61 jfk:2월 129 lga:2월 95

min_risk = high_risk_count.loc[high_risk_count.groupby('origin')['high_risk_count'].idxmin()].reset_index(drop=True)
#ewr:7월 2,jfk:7월 4 lga:8월 5

df_weather
weather_means = df_weather.groupby('origin')[['temp', 'wind_speed', 'wind_gust', 'visib', 'humid', 'precip']].mean().round(2)

# 고위험 조건 정의
df_weather_clean = df_weather.dropna(subset=['wind_speed', 'wind_gust', 'visib']).copy()
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['visib'] < 1)
)

# 공항별 고위험 조건 발생 건수
high_risk_by_origin = df_weather_clean[df_weather_clean['high_risk']].groupby('origin').size().reset_index(name='high_risk_count')
monthly_risk = df_weather_clean[df_weather_clean['high_risk']].groupby(['origin', 'month']).size().reset_index(name='risk_count')

# 1. 강수량 기준 조건 필터링
rain_risk = df_weather[
    (df_weather['precip'].notna()) &
    (df_weather['precip'] >= 0.3)
]

# 2. 공항별 위험 조건 발생 횟수 카운트
rain_risk_by_origin = rain_risk['origin'].value_counts().reset_index()
rain_risk_by_origin.columns = ['origin', 'rain_risk_count']

rain_risk_by_origin
#ewr:25,lga:18,jfk:14

#공항별 시간대 위험조건 발생

# 위험 조건 정의
gust_diff = df_weather.dropna(subset=['wind_gust', 'wind_speed'])
gust_risk = gust_diff[(gust_diff['wind_gust'] - gust_diff['wind_speed']) >= 10]
visib_risk = df_weather[(df_weather['visib'].notna()) & (df_weather['visib'] < 1)]
rain_risk = df_weather[(df_weather['precip'].notna()) & (df_weather['precip'] >= 0.3)]

# 위험 조건 통합
high_risk = pd.concat([gust_risk, visib_risk, rain_risk]).drop_duplicates()

# 공항별 시간대별 위험 발생 수
risk_by_origin_hour = high_risk.groupby(['origin', 'hour']).size().reset_index(name='risk_count')
risk_by_origin_hour

max_risk = risk_by_origin_hour.loc[
    risk_by_origin_hour.groupby('origin')['risk_count'].idxmax()
]
#ewr:13시 55,jfk:16시 32,lga:16시 46

min_risk = risk_by_origin_hour.loc[
    risk_by_origin_hour.groupby('origin')['risk_count'].idxmin()
]
#ewr:3시,12 jfk:23시,16 lga:4시 19

high_risk['time_hour'] = pd.to_datetime(high_risk['time_hour'], errors='coerce')

# 월별 고위험 조건 발생 건수 계산
high_risk['month'] = high_risk['time_hour'].dt.month
month_counts = high_risk['month'].value_counts()
most_risky_month = month_counts.idxmax()
high_risk_month = high_risk[high_risk['month'] == most_risky_month]
risk_by_origin_hour_month = high_risk_month.groupby(['origin', 'hour']).size().reset_index(name='risk_count')
most_common_month = high_risk['time_hour'].dt.month.value_counts().idxmax()
# 최대 시간대
max_hour_month = risk_by_origin_hour_month.loc[
    risk_by_origin_hour_month.groupby('origin')['risk_count'].idxmax()
]

# 최소 시간대
min_hour_month = risk_by_origin_hour_month.loc[
    risk_by_origin_hour_month.groupby('origin')['risk_count'].idxmin()
]
df_weather['time_hour'] = pd.to_datetime(df_weather['time_hour'], errors='coerce')
df_weather['date'] = df_weather['time_hour'].dt.date
daily_risk = high_risk_all.groupby('date').size()


#기상 조건들간의 상관관계 분석 
#어느게 안 좋아지면 어떤것도 안 좋아진다?

#공항별 고위험 조건 유사/차이 분석

# 시간 흐름에 따른 
# 위험 조건 누적 분석 
# 시간 단위다르게? 월,주,일별,시간대별 발생 
#특정 시즌에 맞춰 항공사 지연대응 

#지연이 발생하는 범위를 높음 매우높음 중간까지
#그거에 따라 지연이 발생하는 공항이 달라지고
#월이 달라진다

#하루 중 여러 번 같은 위험조건 반복하거나
#서로 다른 조건들이 발생하는거? 해서 
#해당일자 비행편수를 조정?하던가

