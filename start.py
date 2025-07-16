# conda env export > environment.yml
# conda env create -f environment.yml
import pandas as pd
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather
df_flights.shape # 3 해진 희재 소영
df_airlines.shape # 16
df_airports.shape # 1458 -a 우영
df_planes.shape # 3322 -a 우영
df_weather.shape # 26115 -b 유진
# 예시: 항공편 데이터 확인
print(df_flights.head())

# 오래된 것일수록 비행시간이 늘어나는가?
# 딜레이타임 원인 분석  

# --- 딜레이가 덜 되는 항공사 or 공항
# --- 정시운항이 잘 되는지
# --- 딜레이가 가자 잘 일어나는 패턴-주기
# --- 항공편이 많을수록 딜레이가 자주 일어나는지 >> 공항별 항공편 수
# --- 항공사 별 비행기 수 / 좌석 수
# --- 위도 경도 별 날씨 특징 파악해서 딜레이
# --- 가징 크리티컬하게 영향을 주는 날씨의 특징 
# --- 역풍인지 아닌지를 체크해서 비행시간의 차이를 체크해보자
# --- 성수기일때 실제로 딜레이가 많이 이루어지는지 -- 덜 딜레이되는 최적의 항공사 찾기
# --- 공항마다 딜레이가 가장 잘 되는 시간대

# 늦게 도착하는 3가지 경우를 분석하자
# 1. 일찍 출발해서 2. 정시 출발 3. 늦게 출발
# 데이터프레임 3개로 나누어서 우선 분석
# 정시 출발의 범위를 우선적으로 설정 

df_planes.en
df_weather
df_airlines
df_airports
df_flights['dep_delay'].describe() # 출발 딜레이
df_flights['arr_delay'].describe() # 도착 딜레이
df_flights['dep_delay'].describe()
even_start = df_flights[df_flights['dep_delay']==0] # 정시출발이 1만6천
early_start = df_flights[df_flights['dep_delay']<0] # 일찍 출발 18만3천 10분 일찍 출발까지는 정상 출발로 인정
late_start = df_flights[df_flights['dep_delay']>0] # 연착 12만

df_flights