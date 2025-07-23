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
# 정리
# 전체 항공 데이터 = flights_cleanded
# 15분 이상 출발 지연 데이터 = flights_delay

#################################################################################################
## 전체적인 데이터 탐색

### 항공사별 비교
# 각 항공사별 항공편의 총합을 통해 1,2,3 순위 설정
flight_counts = flights_cleaned['carrier'].value_counts().reset_index() # UA B6 EV

# 3사의 공항에 따른 지연율
# 분석 대상 항공사
selected_carriers = ['UA', 'B6', 'EV']
flights_cleaned = flights_cleaned[flights_cleaned['carrier'].isin(selected_carriers)]

# 3사 지연율 및 평균 계산 > summary_big3를 통해서 3사 중 EV의 지연율이 특출나게 높은 것을 확인할 수 있음
summary_big3 = flights_cleaned.groupby('carrier').apply(
    lambda g: pd.Series({
        'total_flights': len(g),
        'delayed_flights': (g['dep_delay'] >= 15).sum(),
        'delay_rate (%)': round((g['dep_delay'] >= 15).mean() * 100, 2),
        'avg_delay (min)': round(g[g['dep_delay'] >= 15]['dep_delay'].mean(), 2)
    })
).reset_index()
summary_big3 = summary_big3.sort_values("total_flights", ascending=False)
summary_big3 

#################################################################################################
## EV만의 특징 분석 1. 월별 분석 2. 제조사 분석 3. 

# 월별 분석
UA_total = flights_cleaned[flights_cleaned['carrier']=='UA']
B6_total = flights_cleaned[flights_cleaned['carrier']=='B6']
EV_total = flights_cleaned[flights_cleaned['carrier']=='EV']

# 전체 데이터 통계
total_group = flights_cleaned.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# 전체 결항 데이터 통계 
delay_group = flights_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# 전체 데이터 통계 데이터 total_group에 월 별 지연 비율 추가
total_group['delay_ratio'] = delay_group['count'] / total_group['count']

# UA 전체 데이터 통계 
UA_total_group = UA_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# UA 결항 데이터 통계 
UA_delay = UA_total[UA_total['dep_delay']>15].reset_index(drop=True)
UA_delay_group = UA_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# UA 전체 데이터 통계 데이터 UA_total_group에 월 별 지연 비율 추가
UA_total_group['delay_ratio'] = UA_delay_group['count'] / UA_total_group['count']

# B6 전체 데이터 통계 
B6_total_group = B6_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# B6 결항 데이터 통계 
B6_delay = B6_total[B6_total['dep_delay']>15].reset_index(drop=True)
B6_delay_group = B6_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# B6 전체 데이터 통계 데이터 B6_total_group에 월 별 지연 비율 추가
B6_total_group['delay_ratio'] = B6_delay_group['count'] / B6_total_group['count']

# EV 전체 데이터 통계 
EV_total_group = EV_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# EV 결항 데이터 통계 
EV_delay = EV_total[EV_total['dep_delay']>15].reset_index(drop=True)
EV_delay_group = EV_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
# EV 전체 데이터 통계 데이터 EV_total_group에 월 별 지연 비율 추가
EV_total_group['delay_ratio'] = EV_delay_group['count'] / EV_total_group['count']

# 1월~12월 시각화
# subplot 설정
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Monthly Flight Count and Delay Ratio by Airline', fontsize=18)

# 데이터와 라벨 매핑
airline_data = {
    'EV': EV_total_group,
    'B6': B6_total_group,
    'UA': UA_total_group,
    'Total': total_group
}

# subplot 위치 매핑
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

# 각 subplot에 그래프 그리기
for (label, df), (i, j) in zip(airline_data.items(), positions):
    ax1 = axes[i][j]

    # 막대 그래프
    sns.barplot(data=df, x='month', y='count', color='skyblue', ax=ax1)
    ax1.set_ylabel('Flight Count')
    ax1.set_xlabel('Month')
    ax1.set_title(f'{label} - Flight Count and Delay Ratio')

    # 선 그래프 (지연 비율)
    x_coords = ax1.get_xticks()
    ax2 = ax1.twinx()
    ax2.plot(x_coords, df['delay_ratio'], color='red', marker='o', label='Delay Ratio')
    ax2.set_ylabel('Delay Ratio')
    ax2.legend(loc='upper right')

# 레이아웃 조정
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show() # 1~6월 지연 비율이 높은 것을 확인할 수 있음

















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
# 시간순 정렬
ev_schedule = ev_schedule.sort_values(['tailnum', 'month_day_time'])


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