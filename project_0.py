import pandas as pd
import nycflights13 as flights
import numpy as np
# 시각화 라이브러리리
import seaborn as sns
import matplotlib.pyplot as plt

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# 결측치 처리 (결항)
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned = flights_cleaned.reset_index(drop=True)
flights_cleaned
# 1,2,3등 항공사별로 데이터프레임 분리
UA_total = flights_cleaned[flights_cleaned['carrier']=='UA']
B6_total = flights_cleaned[flights_cleaned['carrier']=='B6']
EV_total = flights_cleaned[flights_cleaned['carrier']=='EV']
#######################################################################
# 전체 항공편
# 전체 데이터 통계 
total_group = flights_cleaned.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
total_group
# UA 결항 데이터 통계 12109개
flights_delay = flights_cleaned[flights_cleaned['dep_delay']>15].reset_index(drop=True)
delay_group = flights_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
delay_group
# UA 전체 데이터 통계 데이터 UA_total_group에 월 별 지연 비율 추가
total_group['delay_ratio'] = delay_group['count'] / total_group['count']
total_group

# UA 전체 데이터 통계 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 막대 그래프
sns.barplot(data=total_group, x='month', y='count', color='skyblue', ax=ax1)
ax1.set_ylabel('Flight Count')
ax1.set_title('flight Count and Delay Ratio')

# x축의 실제 위치 (카테고리형 bar 위치) 가져오기
x_coords = ax1.get_xticks()  # [0, 1, 2, ..., 11]

# 선 그래프 (bar 중심에 맞게 x 좌표 지정)
ax2 = ax1.twinx()
ax2.plot(x_coords, total_group['delay_ratio'], color='red', marker='o', label='Delay Ratio')
ax2.set_ylabel('Delay Ratio')
ax2.legend(loc='upper right')
plt.show()
# 지연 횟수 및 평균 시각화
# 횟수
sns.barplot(data=delay_group, x='month', y='count', color='skyblue', label='UA_delay_count')
plt.title('UA_Delay_count')
plt.ylabel('count')
plt.legend()
plt.show()
# 평균
sns.barplot(data=delay_group, x='month', y='mean', color='skyblue', label='UA_delay_mean')
plt.title('UA_Delay_mean')
plt.ylabel('mean')
plt.legend()
plt.show()

#########################################################################
# UA 항공사
# UA 전체 데이터 통계 
UA_total_group = UA_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
UA_total_group
# UA 결항 데이터 통계 12109개
UA_delay = UA_total[UA_total['dep_delay']>15].reset_index(drop=True)
UA_delay_group = UA_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
UA_delay_group
# UA 전체 데이터 통계 데이터 UA_total_group에 월 별 지연 비율 추가
UA_total_group['delay_ratio'] = UA_delay_group['count'] / UA_total_group['count']
UA_total_group

# UA 전체 데이터 통계 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 막대 그래프
sns.barplot(data=UA_total_group, x='month', y='count', color='skyblue', ax=ax1)
ax1.set_ylabel('Flight Count')
ax1.set_title('UA flight Count and Delay Ratio')

# x축의 실제 위치 (카테고리형 bar 위치) 가져오기
x_coords = ax1.get_xticks()  # [0, 1, 2, ..., 11]

# 선 그래프 (bar 중심에 맞게 x 좌표 지정)
ax2 = ax1.twinx()
ax2.plot(x_coords, UA_total_group['delay_ratio'], color='red', marker='o', label='Delay Ratio')
ax2.set_ylabel('Delay Ratio')
ax2.legend(loc='upper right')
plt.show()
# 지연 횟수 및 평균 시각화
# 횟수
sns.barplot(data=UA_delay_group, x='month', y='count', color='skyblue', label='UA_delay_count')
plt.title('UA_Delay_count')
plt.ylabel('count')
plt.legend()
plt.show()
# 평균
sns.barplot(data=UA_delay_group, x='month', y='mean', color='skyblue', label='UA_delay_mean')
plt.title('UA_Delay_mean')
plt.ylabel('mean')
plt.legend()
plt.show()
########################################################################

# B6 항공사
# B6 전체 데이터 통계 
B6_total_group = B6_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
B6_total_group
# B6 결항 데이터 통계 12109개
B6_delay = B6_total[B6_total['dep_delay']>15].reset_index(drop=True)
B6_delay_group = B6_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
B6_delay_group
# B6 전체 데이터 통계 데이터 UA_total_group에 월 별 지연 비율 추가
B6_total_group['delay_ratio'] = B6_delay_group['count'] / B6_total_group['count']
B6_total_group

# B6 전체 데이터 통계 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 막대 그래프
sns.barplot(data=B6_total_group, x='month', y='count', color='skyblue', ax=ax1)
ax1.set_ylabel('Flight Count')
ax1.set_title('B6 flight Count and Delay Ratio')

# x축의 실제 위치 (카테고리형 bar 위치) 가져오기
x_coords = ax1.get_xticks()  # [0, 1, 2, ..., 11]

# 선 그래프 (bar 중심에 맞게 x 좌표 지정)
ax2 = ax1.twinx()
ax2.plot(x_coords, B6_total_group['delay_ratio'], color='red', marker='o', label='Delay Ratio')
ax2.set_ylabel('Delay Ratio')
ax2.legend(loc='upper right')
plt.show()
# 지연 횟수 및 평균 시각화
# 횟수
sns.barplot(data=B6_delay_group, x='month', y='count', color='skyblue', label='B6_delay_count')
plt.title('B6_Delay_count')
plt.ylabel('count')
plt.legend()
plt.show()
# 평균
sns.barplot(data=B6_delay_group, x='month', y='mean', color='skyblue', label='B6_delay_mean')
plt.title('B6_Delay_mean')
plt.ylabel('mean')
plt.legend()
plt.show()
##############################################################
# EV 항공사
# EV 전체 데이터 통계 
EV_total_group = EV_total.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
EV_total_group
# UA 결항 데이터 통계 12109개
EV_delay = EV_total[EV_total['dep_delay']>15].reset_index(drop=True)
EV_delay_group = EV_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
EV_delay_group
# UA 전체 데이터 통계 데이터 UA_total_group에 월 별 지연 비율 추가
EV_total_group['delay_ratio'] = EV_delay_group['count'] / EV_total_group['count']
EV_total_group

# UA 전체 데이터 통계 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 막대 그래프
sns.barplot(data=EV_total_group, x='month', y='count', color='skyblue', ax=ax1)
ax1.set_ylabel('Flight Count')
ax1.set_title('EV flight Count and Delay Ratio')

# x축의 실제 위치 (카테고리형 bar 위치) 가져오기
x_coords = ax1.get_xticks()  # [0, 1, 2, ..., 11]

# 선 그래프 (bar 중심에 맞게 x 좌표 지정)
ax2 = ax1.twinx()
ax2.plot(x_coords, EV_total_group['delay_ratio'], color='red', marker='o', label='Delay Ratio')
ax2.set_ylabel('Delay Ratio')
ax2.legend(loc='upper right')
plt.show()
# 지연 횟수 및 평균 시각화
# 횟수
sns.barplot(data=EV_delay_group, x='month', y='count', color='skyblue', label='UA_delay_count')
plt.title('EV_Delay_count')
plt.ylabel('count')
plt.legend()
plt.show()
# 평균
sns.barplot(data=EV_delay_group, x='month', y='mean', color='skyblue', label='UA_delay_mean')
plt.title('EV_Delay_mean')
plt.ylabel('mean')
plt.legend()
plt.show()

# EV가 상반기 지연비율이 굉장히 높음















