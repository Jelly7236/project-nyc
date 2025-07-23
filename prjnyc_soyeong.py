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

flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])

# 항공사별 총 항공편 수 계산
flight_counts = flights_cleaned['carrier'].value_counts().reset_index()

# 상위 3개 항공사의 노선 길이 비교
top_3_carriers = flight_counts['carrier'].head(3).tolist()

# 1. UA , 57782
flights_cleaned['carrier']=='UA' # 57782

# distance, air_time 확인
UA_flight=flights_cleaned.loc[flights_cleaned['carrier']=='UA',['carrier','distance','air_time']]

UA_flight.sort_values(by=['distance','air_time'], ascending=[False,True])
# distance는 내림차순, air_time은 오름차순으로 정렬( 동일한 거리에서 시간이 짧을 수록 비행 good)

UA_flight.describe()

# 2. B6, 54049
flights_cleaned['carrier']=='B6' # 54049

# distance, air_time 확인
B6_flight=flights_cleaned.loc[flights_cleaned['carrier']=='B6',['carrier','distance','air_time']]

B6_flight.sort_values(by=['distance','air_time'], ascending=[False,True])

B6_flight.describe()

# 3. EV, 51108
flights_cleaned['carrier']=='EV' # 51108

# distance, air_time 확인
EV_flight=flights_cleaned.loc[flights_cleaned['carrier']=='EV',['carrier','distance','air_time']]

EV_flight.sort_values(by=['distance','air_time'], ascending=[False,True])


EV_flight.describe()

'''
United Airline
700 mile 미만 -> short 
700 mile 이상 3000 mile 미만 -> medium
3000 mile 이상 -> long
'''
# 상위 항공사 3개를 합친 새로운 DataFrame 생성
top3_flights = pd.concat([UA_flight, B6_flight, EV_flight], ignore_index=True)

# 길이 (mile) 에 따른 기준 생성
def categorize_distance(mile):
    if mile < 700:
        return 'short'
    elif mile < 3000:
        return 'medium'
    else:
        return 'long'
    
# length라는 새 column을 만들어 거리를 비교    
top3_flights['length'] = top3_flights['distance'].apply(categorize_distance)    

# pivot table 생성
pivot_flight = pd.pivot_table(
    top3_flights,
    index='carrier',
    columns='length',
    values='distance',      
    aggfunc='count',
    fill_value=0
).reset_index()
pivot_flight.columns.name = None
pivot_flight = pivot_flight[['carrier', 'short', 'medium', 'long']]
pivot_flight

######## top3 항공사의 노선거리 분석 끝 ########



### EV 항공사의 기체 별 회전율 시작 ###


# 월,일,시간,분 으로 나눠진 datetime column을 새로 만듦
df_flights['month_day_time'] = pd.to_datetime({
    'year': df_flights['year'],
    'month': df_flights['month'],
    'day': df_flights['day'],
    'hour': df_flights['hour'],
    'minute': df_flights['minute']
})

# EV 항공사 중에서 tailnum, 출발 시간, 날짜 정보 추출
ev_schedule = df_flights[df_flights['carrier'] == 'EV'][['tailnum', 'month_day_time']].dropna()

# 1. datetime 형식으로 먼저 변환
ev_schedule['month_day_time'] = pd.to_datetime(ev_schedule['month_day_time'])

# 2. tailnum 기준으로 시간 차이(diff) 계산 후, 이를 time_gap 이라는 새로운 column으로 추가

ev_schedule['time_gap'] = ev_schedule.groupby('tailnum')['month_day_time'].diff()

# 값이 NaT인 경우, 
# 앞쪽 값들 (NaT): 해당 tailnum 그룹에서 첫 비행 → 비교 대상 없음 → NaT

# 간격을 시간(hour) 단위로 변경
ev_schedule['gap_hours'] = ev_schedule['time_gap'].dt.total_seconds() / 3600


# 3. 결과 확인
ev_schedule['time_gap']
print(ev_schedule.head(10))

# 4. 값 확인
schedule_summary = ev_schedule.groupby('tailnum')['gap_hours'].agg(['mean', 'min', 'count']).reset_index().sort_values('count',ascending=False)

'''
gap_hours에 대해서, 

'mean': 평균 비행 간격 (단위: 시간)

'min': 가장 짧은 간격

'count': 비행 횟수(정확히는 gap_hours 값이 있는 횟수 = 첫 비행 제외

'''

schedule_summary['count'].describe()

schedule_summary.dropna()

schedule_summary.columns = ['tailnum', 'avg_gap_hr', 'min_gap_hr', 'flight_count']

# 평균 비행 간격이 짧은 상위 5개 항공기 + 좌석수 보기
top_planes = schedule_summary.sort_values('avg_gap_hr').head(5)

plt.figure(figsize=(10, 5))
sns.barplot(data=top_planes, x='tailnum', y='avg_gap_hr')
plt.xticks(rotation=45)
plt.title('The 5 EV aircraft with the shortest average flight intervals')
plt.xlabel('Tail Number')
plt.ylabel('Average Gap (hours)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 비행기의 좌석 수도 함께 보고 싶다면,

top_planes_with_seats = pd.merge(
    top_planes,
    df_planes[['tailnum', 'seats']],  # 필요한 열만 선택
    on='tailnum',
    how='left'  # 좌측 기준으로 결합 (top_planes 기준)
)
plt.figure(figsize=(10, 5))
sns.barplot(data=top_planes_with_seats, x='tailnum', y='avg_gap_hr')
plt.xticks(rotation=45)
plt.title('The 5 EV aircraft with the shortest average flight intervals')
plt.xlabel('Tail Number')
plt.ylabel('Average Gap (hours)')
plt.grid(True)

# 막대 위에 좌석 수 표시
for i, row in top_planes_with_seats.iterrows():
    if pd.notna(row['seats']):
        plt.text(
            x=i,
            y=row['avg_gap_hr'] + 0.2,  # 막대 위 약간 띄우기
            s=f"{int(row['seats'])} seats",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

plt.tight_layout()
plt.show()

# 시각화는 좀 더 예뿌게 부탁드립니다 ^_^*