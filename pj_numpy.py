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
#플라이트 결항 지연 찾기
#출발지연 결항 15분이상 지연된 항공편 보잉사

# 예시: 항공편 데이터 확인
print(df_flights.head())
df_flights
df_airports
df_planes['model']('BOEING')
df_weather
B=df_planes[df_planes['model'] == 'BOEING']
B
# 1. 'engine'이 'Turbo-fan'인 항공기만 필터링
turbofan_planes = df_planes[df_planes['engine'].str.contains('Turbo-jet', case=False, na=False)]

# 2. 제조년도 열이 'year'라고 가정하고 최대/최소 구하기
min_year = turbofan_planes['year'].min()
max_year = turbofan_planes['year'].max()

print(f"터보젯 항공기의 제조년도 범위: {min_year} ~ {max_year}")
df_planes['engines']['4']
df_flights
df_planes[df_planes['model'].str.contains('BOEING', na=False)]
def count_daily_flights(df_flights, tailnum, year, month, day):
    result = df_flights[
        (df_flights['tailnum'] == tailnum) &
        (df_flights['year'] == year) &
        (df_flights['month'] == month) &
        (df_flights['day'] == day)
    ]
    return result.shape[0], result
BB = df_planes[df_planes['model'] == 'BOEING']
B = df_planes[df_planes['manufacturer'] == 'BOEING']
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

# EV 항공사 데이터만 추출
ev_flights = df_flights[df_flights['carrier'] == 'EV']

# origin(출발 공항), month별 운항 횟수 집계
ev_monthly_origin = ev_flights.groupby(['month', 'origin']).size().reset_index(name='flight_count')

# 각 월별로 가장 많이 이용한 공항 추출
most_used_origin_by_month = ev_monthly_origin.loc[
    ev_monthly_origin.groupby('month')['flight_count'].idxmax()
].reset_index(drop=True)

# 결과 확인
print(most_used_origin_by_month)

import pandas as pd

# 1. 필요한 조건만 필터링
gust_risk = df_weather.dropna(subset=['wind_gust', 'wind_speed'])
gust_risk = gust_risk[(gust_risk['wind_gust'] - gust_risk['wind_speed']) >= 10]

visib_risk = df_weather[(df_weather['visib'].notna()) & (df_weather['visib'] < 1)]

# 2. 두 조건을 통합 (강수량 제외!)
high_risk_weather = pd.concat([gust_risk, visib_risk]).drop_duplicates()

# 3. 시간 처리
high_risk_weather['time_hour'] = pd.to_datetime(high_risk_weather['time_hour'], errors='coerce')
high_risk_weather['month'] = high_risk_weather['time_hour'].dt.month
high_risk_weather = high_risk_weather[high_risk_weather['month'].between(1, 6)]
# 4. 공항 × 월별 위험 조건 발생 수
monthly_high_risk = high_risk_weather.groupby(['origin', 'month']).size().reset_index(name='risk_count')

# 5. 결과 보기
print(monthly_high_risk)


# 1. 결측치 제거
df_flights_clean = df_flights.dropna(subset=['origin', 'carrier', 'month'])

# 2. 상위 3개 항공사만 필터링 & 1~6월만 선택
top3_carriers = ['UA', 'B6', 'EV']
filtered = df_flights_clean[
    (df_flights_clean['carrier'].isin(top3_carriers)) &
    (df_flights_clean['month'].between(1, 6))
]

# 3. 월별 항공사 × 공항별 운항 횟수 집계
monthly_usage = filtered.groupby(['carrier', 'month', 'origin']).size().reset_index(name='flight_count')

# 4. 보기 좋게 피벗 형태로 변환
pivot_table = monthly_usage.pivot_table(
    index=['carrier', 'month'],
    columns='origin',
    values='flight_count',
    fill_value=0
).astype(int)

# 결과 출력
import ace_tools as tools; tools.display_dataframe_to_user(name="1~6월 항공사별 공항 이용 현황", dataframe=pivot_table)

#EV는 LGA,EWR을 동시에 집중 운항해 
# 두 공항의 날씨 위험에 이중 노출
#UA는 EWR 중심이지만 기종 크기상 지연에 덜 민감

import pandas as pd
import matplotlib.pyplot as plt

# flights_cleaned = 결측치 제거한 flights 데이터
top3_carriers = ['EV', 'UA', 'B6']
filtered = flights_cleaned[flights_cleaned['carrier'].isin(top3_carriers)]

# 지연 여부 플래그
filtered['delayed_15'] = filtered['dep_delay'] >= 15

# 월별 지연율 집계
delay_summary = filtered.groupby(['carrier', 'month'])['delayed_15'].agg(['mean', 'count', 'sum']).reset_index()
delay_summary.columns = ['carrier', 'month', 'delay_rate', 'total_flights', 'delayed_flights']

# 1~6월만 필터
delay_summary = delay_summary[delay_summary['month'] <= 6]

# 시각화
plt.figure(figsize=(12, 6))
for carrier in top3_carriers:
    subset = delay_summary[delay_summary['carrier'] == carrier]
    plt.plot(subset['month'], subset['delay_rate'], marker='o', label=carrier)

plt.title('1~6월 항공사별 출발 지연율 (15분 이상)')
plt.xlabel('월')
plt.ylabel('지연율')
plt.xticks(range(1, 7))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 데이터 불러오기
# 상위 3개 항공사만 필터링
top3_carriers = ['EV', 'UA', 'B6']
filtered = flights_cleaned[flights_cleaned['carrier'].isin(top3_carriers)].copy()

# 지연 여부 플래그 추가
filtered['delayed_15'] = filtered['dep_delay'] >= 15

# 월별 지연율 계산
delay_summary = filtered.groupby(['carrier', 'month'])['delayed_15'].agg(['mean', 'count', 'sum']).reset_index()
delay_summary.columns = ['carrier', 'month', 'delay_rate', 'total_flights', 'delayed_flights']

# 시각화
plt.figure(figsize=(12, 6))
for carrier in top3_carriers:
    subset = delay_summary[delay_summary['carrier'] == carrier]
    plt.plot(subset['month'], subset['delay_rate'], marker='o', label=carrier)

plt.title('1~12월 항공사별 출발 지연율 (15분 이상)')
plt.xlabel('월')
plt.ylabel('지연율')
plt.xticks(range(1, 13))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 1. flights와 planes 병합 (모델 정보 추가)
merged = pd.merge(
    flights_cleaned,
    df_planes[['tailnum', 'model']],
    on='tailnum',
    how='inner'
)

# 2. EV 항공사의 기종 수 확인
ev_models = merged[merged['carrier'] == 'EV']['model'].nunique()
print(f"EV 항공사가 운용한 기종 수: {ev_models}")

# 3. 항공사별 기종별 운항량과 지연율 계산
merged['delayed_15'] = merged['dep_delay'] >= 15
model_stats = merged.groupby(['carrier', 'model']).agg(
    total_flights=('flight', 'count'),
    delayed_flights=('delayed_15', 'sum'),
    delay_rate=('delayed_15', 'mean')
).reset_index()

# 4. EV 항공사만 필터링하여 확인
ev_model_stats = model_stats[model_stats['carrier'] == 'EV'].sort_values(by='total_flights', ascending=False)
print(ev_model_stats)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(ev_model_stats['total_flights'], ev_model_stats['delay_rate'], s=100)

for _, row in ev_model_stats.iterrows():
    plt.text(row['total_flights'], row['delay_rate'], row['model'], fontsize=9)

plt.xlabel("운항 횟수")
plt.ylabel("지연율")
plt.title("EV 항공사 기종별 운항량 vs 지연율")
plt.grid(True)
plt.tight_layout()
plt.show()

df_flights = flights.flights
df_planes = flights.planes

# 결측치 제거
flights_cleaned = df_flights.dropna(subset=['dep_time', 'dep_delay', 'arr_time', 'arr_delay', 'tailnum', 'air_time'])

# 병합
merged_df = pd.merge(flights_cleaned, df_planes[['tailnum', 'model']], on='tailnum', how='inner')

# 중복 제거 후 항공사별 비행기 대수 확인
unique_planes = merged_df[['carrier', 'tailnum']].drop_duplicates()
plane_counts = unique_planes[unique_planes['carrier'].isin(['EV', 'UA', 'B6'])].groupby('carrier').size().reset_index(name='num_planes')

print(plane_counts)

# 지연된 항공편만 필터링
merged_df['delayed_15'] = merged_df['dep_delay'] >= 15
delayed_flights = merged_df[merged_df['carrier'].isin(top3_carriers) & merged_df['delayed_15']]

# 항공사별 지연 항공편 수
delay_counts = delayed_flights.groupby('carrier').size().reset_index(name='num_delayed_flights')
delay_counts


# 병합: 비행기 대수 + 지연 항공편 수
result = pd.merge(plane_counts, delay_counts, on='carrier')
result['delays_per_plane'] = result['num_delayed_flights'] / result['num_planes']

print(result)

import pandas as pd
from nycflights13 import flights

# 1. 필요한 데이터 로딩
df_flights = flights.flights

# 2. 필요한 항공사만 필터링하고, 1~6월 조건 적용
top3_carriers = ['UA', 'B6', 'EV']
filtered = df_flights[
    (df_flights['carrier'].isin(top3_carriers)) &
    (df_flights['month'].between(1, 6)) &
    (df_flights['origin'].notna())
]

# 3. 그룹핑: 공항(origin), 항공사(carrier)별 운항 횟수 집계
airport_monthly_flight_counts = filtered.groupby(['origin', 'carrier']).size().reset_index(name='flight_count')

# 결과 보기
print(airport_monthly_flight_counts)

df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# 2. 대상 항공사 및 월 필터링
top3_carriers = ['UA', 'B6', 'EV']
filtered = df_flights[
    (df_flights['carrier'].isin(top3_carriers)) &
    (df_flights['month'].between(1, 6)) &
    (df_flights['origin'].notna())
]

# 3. 공항(origin)별, 항공사별 운항량 집계
airport_flight_counts = filtered.groupby(['carrier', 'origin']).size().reset_index(name='flight_count')

# 4. 보기 쉽게 pivot table로 정리
pivot_table = airport_flight_counts.pivot(index='carrier', columns='origin', values='flight_count').fillna(0).astype(int)

# 5. 결과 출력
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('1~6월 항공사별 공항 이용 운항량')
plt.xlabel('출발 공항(origin)')
plt.ylabel('항공사(carrier)')
plt.tight_layout()
plt.show()

flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned 
# 결측치 제거
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])

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

import pandas as pd
import nycflights13 as flights

# 2. 데이터 불러오기
df_weather = flights.weather

# 3. 결측치 제거 및 고위험 조건 정의
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib']).copy()
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['visib'] < 1)
)

# 🔧 4. time_hour 열을 datetime 형식으로 변환
df_weather_clean['time_hour'] = pd.to_datetime(df_weather_clean['time_hour'], errors='coerce')

# 5. 월(month) 추출 및 1~6월 필터링
df_weather_clean['month'] = df_weather_clean['time_hour'].dt.month
df_weather_clean = df_weather_clean[df_weather_clean['month'].between(1, 6)]

# 6. 전체 관측수 및 고위험 조건 발생수 계산
total_obs = df_weather_clean.groupby(['origin', 'month']).size().reset_index(name='total_obs')
high_risk_obs = df_weather_clean[df_weather_clean['high_risk']].groupby(['origin', 'month']).size().reset_index(name='high_risk_obs')

# 7. 병합 및 비율 계산
risk_table = pd.merge(total_obs, high_risk_obs, on=['origin', 'month'], how='left').fillna(0)
risk_table['risk_ratio'] = risk_table['high_risk_obs'] / risk_table['total_obs']

# 8. 결과 출력
import ace_tools as tools; tools.display_dataframe_to_user(name="1~6월 공항별 고위험 날씨 비율", dataframe=risk_table)

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

import pandas as pd
import numpy as np

# 1. 고위험 조건 정의를 위한 결측치 제거 및 datetime 처리
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib', 'time_hour']).copy()
df_weather_clean['time_hour'] = pd.to_datetime(df_weather_clean['time_hour'], errors='coerce')
df_weather_clean['month'] = df_weather_clean['time_hour'].dt.month

# 2. 고위험 조건 플래그 생성
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['visib'] < 1)
)

# 3. 월별 전체 관측 수
total_by_month = df_weather_clean.groupby('month').size().reset_index(name='total_obs')

# 4. 월별 고위험 조건 발생 수
high_risk_by_month = df_weather_clean[df_weather_clean['high_risk']].groupby('month').size().reset_index(name='high_risk_obs')

# 5. 병합 및 지연율 계산
month_risk_ratio = pd.merge(total_by_month, high_risk_by_month, on='month', how='left').fillna(0)
month_risk_ratio['risk_ratio'] = month_risk_ratio['high_risk_obs'] / month_risk_ratio['total_obs']

# 6. 출력
print(month_risk_ratio)

# 데이터 로드
df_weather = flights.weather

# 결측치 제거 및 datetime 처리
df_weather_clean = df_weather.dropna(subset=['wind_gust', 'wind_speed', 'visib', 'time_hour']).copy()
df_weather_clean['time_hour'] = pd.to_datetime(df_weather_clean['time_hour'], errors='coerce')
df_weather_clean['month'] = df_weather_clean['time_hour'].dt.month

# 고위험 조건 플래그
df_weather_clean['high_risk'] = (
    (df_weather_clean['wind_gust'] >= 30) |
    (df_weather_clean['wind_speed'] >= 20) |
    (df_weather_clean['visib'] < 1)
)

# 공항 × 월별 전체 관측 수
total_by_airport_month = df_weather_clean.groupby(['origin', 'month']).size().reset_index(name='total_obs')

# 공항 × 월별 고위험 조건 발생 수
high_risk_by_airport_month = df_weather_clean[df_weather_clean['high_risk']].groupby(['origin', 'month']).size().reset_index(name='high_risk_obs')

# 병합 및 비율 계산
airport_month_risk_ratio = pd.merge(total_by_airport_month, high_risk_by_airport_month, on=['origin', 'month'], how='left').fillna(0)
airport_month_risk_ratio['risk_ratio'] = airport_month_risk_ratio['high_risk_obs'] / airport_month_risk_ratio['total_obs']

# 결과 보기
print(airport_month_risk_ratio)


import matplotlib.pyplot as plt

# 시각화 스타일 설정 (선택 사항)
plt.style.use('seaborn-whitegrid')

# 공항 리스트
airports = airport_month_risk_ratio['origin'].unique()

# 공항별 선 그래프
plt.figure(figsize=(12, 6))

for airport in airports:
    data = airport_month_risk_ratio[airport_month_risk_ratio['origin'] == airport]
    plt.plot(data['month'], data['risk_ratio'], marker='o', label=airport)

# 그래프 꾸미기
plt.title('월별 공항별 날씨 고위험 조건 발생률 (2013)', fontsize=14)
plt.xlabel('월', fontsize=12)
plt.ylabel('고위험 날씨 발생 비율', fontsize=12)
plt.xticks(range(1, 13))
plt.legend(title='공항')
plt.tight_layout()
plt.show()