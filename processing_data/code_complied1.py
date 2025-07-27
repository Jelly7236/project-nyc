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
## EV만의 특징 분석 1. 월별 분석 2. 제조사 분석 3. 비행 시간 비교
###########################################################################################
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

#################################################################################
# 2. 제조사 분석 >> 좌석수가 적은 것을 알 수 있었고 이를 통해 주로 작은 항공기를 운영중을 확인
##################################################################################
# 사용할 carrier만 필터링
big3 = flights_cleaned[flights_cleaned['carrier'].isin(['UA', 'B6', 'EV'])]
# 항공사별 제조사 그룹화
plane_features = (
    big3
    .dropna(subset=['manufacturer', 'seats'])
    .groupby(['carrier', 'manufacturer'])
    .agg(
        avg_seats=('seats', 'mean'),
        avg_engines=('engines', 'mean'),
        count=('tailnum', 'nunique')
    )
    .reset_index()
)
print("▶ 항공사별 제조사 분포:")
print(plane_features)
# 항공사별 좌석수 평균 확인
avg_seats = (
    big3.dropna(subset=['seats'])
    .groupby('carrier')['seats']
    .mean()
    .reset_index()
    .rename(columns={'seats': 'avg_seats'})
)

print("\n▶ 항공사별 평균 좌석 수:")
print(avg_seats)


##############################################################################
# 3. 각 항공사 별 단거리 중거리 장거리 파악 >> EV는 작은 비행기를 운행하는 단거리 항공사이다
###############################################################################

# 1. UA , 57782
# distance, air_time 확인
UA_flight=UA_total[['carrier','distance','air_time']]
UA_flight.sort_values(by=['distance','air_time'], ascending=[False,True])
# distance는 내림차순, air_time은 오름차순으로 정렬( 동일한 거리에서 시간이 짧을 수록 비행 good)
UA_flight.describe()

# 2. B6, 54049
# distance, air_time 확인
B6_flight= B6_total[['carrier','distance','air_time']]
B6_flight.sort_values(by=['distance','air_time'], ascending=[False,True])
B6_flight.describe()

# 3. EV, 51108
# distance, air_time 확인
EV_flight=EV_total[['carrier','distance','air_time']]
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


###########################################################################
# 지연 원인 분석 1. 기체 회전율이 낮음 2. 장거리 위주의 공항 존재
#####################################################################

##################################################################
### 1. 낮은 회전율: EV 항공사의 기체 별 회전율 ###
#################################################################

# EV 항공사 중에서 tailnum, 출발 시간, 날짜 정보 추출
ev_schedule = EV_total[['tailnum', 'month_day_time']].dropna()

# 1. datetime 형식으로 먼저 변환
ev_schedule['month_day_time'] = pd.to_datetime(ev_schedule['month_day_time'])
# 시간순 정렬   # 수정사항 원래는 앞에 TAILNUM 이 있었는데 이럼 정렬 ㄴㄴ
ev_schedule = ev_schedule.sort_values(['month_day_time'])

# 2. tailnum 기준으로 시간 차이(diff) 계산 후, 이를 time_gap 이라는 새로운 column으로 추가
# diff()
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

# 평균 비행 간격이 짧은 상위 10개 항공기 시각화
top_planes = schedule_summary.sort_values('avg_gap_hr').head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_planes, x='tailnum', y='avg_gap_hr')
plt.xticks(rotation=45)
plt.title('평균 운항 간격이 가장 짧은 EV 항공기 10대', fontsize=16, fontweight='bold')
plt.xlabel('항공기 번호 (Tail Number)', fontsize=12)
plt.ylabel('평균 간격 (시간)', fontsize=12)
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

################################################################
# 장거리 위주의 공항이 있음
################################################################
origin_summary = flights_cleaned.groupby('origin')[['distance', 'air_time']].mean().reset_index()
origin_summary 
# jfk 거리가 1222 lga는 872로 평균 거리차이가 있음
# 시간도 171 , 127로 차이가 있음

# 1. 세 항공사만 필터링
big3 = flights_cleaned[flights_cleaned['carrier'].isin(['UA', 'B6', 'EV'])]

# 2. origin과 carrier 기준으로 항공편 수 집계
big3_counts = (
    big3.groupby(['origin', 'carrier'])
    .size()
    .unstack(fill_value=0)  # carrier를 열로, 결측은 0으로 채움
    .reset_index()
)  ####### 이 통계자료는 보여주는거는 고려해봐야할듯
# 원래는 JFK에 UA랑B6가 많고 LGA에는 EV가 가장 많은 것을 보여줄려고했는데
# JFK공항에 EV 항공편이 너무 적은게 문제임 다른건ㄱㅊ

# 각 공항별 3사의 지연율 확인
# 15분 이상 지연된 항공편만 필터링
big3_delay = flights_cleaned[
    (flights_cleaned['carrier'].isin(selected_carriers)) &
    (flights_cleaned['dep_delay'] >= 15)
]

# 결측값 제거 후 평균 출발 지연 시간 계산 (carrier, origin 기준)
airport_delay = (
    big3_delay.dropna(subset=['dep_delay'])
    .groupby(['carrier', 'origin'])['dep_delay']
    .mean()
    .reset_index()
)

# 원하는 carrier 순서로 지정
carrier_order = ['UA', 'B6', 'EV']
airport_delay['carrier'] = pd.Categorical(airport_delay['carrier'], categories=carrier_order, ordered=True)

# 공항 이름 붙이기 (origin code -> 공항명)
airport_names = df_airports[['faa', 'name']].rename(columns={'faa': 'origin', 'name': 'airport_name'})
airport_delay = pd.merge(airport_delay, airport_names, on='origin', how='left')

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(data=airport_delay, x='origin', y='dep_delay', hue='carrier')
plt.title('항공사별 출발 공항 기준 평균 출발 지연 시간 (UA, B6, EV)')
plt.xlabel('출발 공항 코드')
plt.ylabel('평균 출발 지연 시간 (분)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


############################
# 공항, 항공사별 지연된 항공편 수

# 1. 세 항공사만 필터링
selected_carriers = ['UA', 'B6', 'EV']
big3 = flights_cleaned[flights_cleaned['carrier'].isin(selected_carriers)]

# 2. 전체 항공편 수 (carrier + origin 기준)
total_counts = (
    big3.groupby(['carrier', 'origin'])
    .size()
    .reset_index(name='total_flights')
)

# 3. 15분 이상 지연된 항공편만 필터링
big3_delay = big3[big3['dep_delay'] >= 15]

# 4. 지연된 항공편 수 (carrier + origin 기준)
delay_counts = (
    big3_delay.groupby(['carrier', 'origin'])
    .size()
    .reset_index(name='delay_count')
)

# 5. 평균 지연 시간 (기존 코드 유지)
airport_delay = (
    big3_delay.dropna(subset=['dep_delay'])
    .groupby(['carrier', 'origin'])['dep_delay']
    .mean()
    .reset_index()
)

# 6. total_flights와 delay_count를 airport_delay에 merge
airport_delay = pd.merge(airport_delay, total_counts, on=['carrier', 'origin'], how='left')
airport_delay = pd.merge(airport_delay, delay_counts, on=['carrier', 'origin'], how='left')

# 7. 결측값 처리 및 지연 비율 계산
airport_delay['delay_count'] = airport_delay['delay_count'].fillna(0)
airport_delay['delay_ratio'] = airport_delay['delay_count'] / airport_delay['total_flights']

# 8. carrier 순서 정렬
carrier_order = ['UA', 'B6', 'EV']
airport_delay['carrier'] = pd.Categorical(airport_delay['carrier'], categories=carrier_order, ordered=True)

# 9. 공항 이름 붙이기 (origin -> 공항명)
airport_names = df_airports[['faa', 'name']].rename(columns={'faa': 'origin', 'name': 'airport_name'})
airport_delay = pd.merge(airport_delay, airport_names, on='origin', how='left')

# 10. 시각화
# 시각화용 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 서브플롯 생성 (1행 2열)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. 평균 지연 시간 시각화
sns.barplot(
    data=airport_delay,
    x='origin', y='dep_delay', hue='carrier',
    ax=axes[0]
)
axes[0].set_title('출발 공항별 항공사 평균 지연 시간')
axes[0].set_xlabel('출발 공항 코드')
axes[0].set_ylabel('평균 지연 시간 (분)')
axes[0].grid(True, axis='y')

# 2. 지연 비율 시각화
sns.barplot(
    data=airport_delay,
    x='origin', y='delay_ratio', hue='carrier',
    ax=axes[1]
)
axes[1].set_title('출발 공항별 항공사 지연 비율 (15분 이상 지연)')
axes[1].set_xlabel('출발 공항 코드')
axes[1].set_ylabel('지연 비율')
axes[1].set_ylim(0, 0.5)
axes[1].grid(True, axis='y')

# 전체 레이아웃 정리
plt.tight_layout()
plt.show()

