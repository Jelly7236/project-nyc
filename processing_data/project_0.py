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
# 전체 결항 데이터 통계 
flights_delay = flights_cleaned[flights_cleaned['dep_delay']>15].reset_index(drop=True)
delay_group = flights_delay.groupby('month',as_index=False)['dep_delay'].agg(['count','mean'])
delay_group
# 전체 데이터 통계 데이터 total_group에 월 별 지연 비율 추가
total_group['delay_ratio'] = delay_group['count'] / total_group['count']
total_group

# 전체 데이터 통계 시각화
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

############################ 한번에 보기
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()

# ev 데이터 더 살피기

EV_total_group[EV_total_group['month']<=5]['count'].sum()
EV_total_group['count'].sum() # 40퍼센트?


# 공항별 단기 비행의 지연율도 파악해서 적은 쪽으로 주로 배치

flights_cleaned.groupby(['month','day'])[['year','dep_delay']].agg({'year':'count','dep_delay':'mean'})
flights_cleaned.info()

###################################################################3
# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'  # Mac
    except:
        plt.rcParams['font.family'] = 'NanumGothic'  # Linux
        
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지


# 15분 이상 지연된 항공편만 필터링하여 평균 계산
delayed_flights = flights_cleaned[flights_cleaned['dep_delay'] >= 15]

# 월별 데이터 그룹화
monthly_data = flights_cleaned.groupby(['month','day']).agg({
    'year': 'count',  # 총 항공편 수
}).reset_index()

# 15분 이상 지연된 항공편의 평균 지연시간 계산
delayed_avg = delayed_flights.groupby(['month','day'])['dep_delay'].mean().reset_index()
delayed_avg.columns = ['month', 'day', 'avg_delay_15plus']

# 15분 이상 지연된 항공편 수 계산
delayed_count = delayed_flights.groupby(['month','day']).size().reset_index(name='delayed_count')

# 데이터 병합
monthly_data = monthly_data.merge(delayed_avg, on=['month', 'day'], how='left')
monthly_data = monthly_data.merge(delayed_count, on=['month', 'day'], how='left')

# NaN 값을 0으로 채우기 (지연이 없는 날)
monthly_data['avg_delay_15plus'] = monthly_data['avg_delay_15plus'].fillna(0)
monthly_data['delayed_count'] = monthly_data['delayed_count'].fillna(0)

# 지연 비율 계산 (15분 이상 지연 / 전체 항공편)
monthly_data['delay_ratio'] = (monthly_data['delayed_count'] / monthly_data['year']) * 100

# 컬럼명 변경
monthly_data.columns = ['month', 'day', 'total_flights', 'avg_delay_15plus', 'delayed_count', 'delay_ratio']

# 12개 월별 서브플롯 생성 (3개 지표를 보여주기 위해 더 큰 크기)
fig, axes = plt.subplots(4, 3, figsize=(20, 18))
fig.suptitle('월별 항공편 현황 (지연 기준: 15분 이상)', fontsize=18, fontweight='bold')

# 월 이름
month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
               '7월', '8월', '9월', '10월', '11월', '12월']

for i in range(12):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # 해당 월 데이터 필터링
    month_data = monthly_data[monthly_data['month'] == i+1]
    
    # 삼중 y축 생성
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    
    # ax3의 spine을 오른쪽으로 이동
    ax3.spines['right'].set_position(('outward', 60))
    
    # 막대 그래프 (총 항공편 수)
    bars = ax.bar(month_data['day'], month_data['total_flights'], 
                  alpha=0.6, color='lightblue', label='총 항공편 수', width=0.8)
    
    # 선 그래프 (15분 이상 지연의 평균 지연시간)
    # 지연이 있는 날만 표시 (0이 아닌 값만)
    delay_data = month_data[month_data['avg_delay_15plus'] > 0]
    if not delay_data.empty:
        line1 = ax2.plot(delay_data['day'], delay_data['avg_delay_15plus'], 
                        color='red', marker='o', markersize=4, linewidth=2, 
                        label='평균 지연시간(15분+)')
    
    # 선 그래프 (지연 비율)
    line2 = ax3.plot(month_data['day'], month_data['delay_ratio'], 
                    color='orange', marker='s', markersize=3, linewidth=2, 
                    label='지연 비율(%)')
    
    # 축 설정
    ax.set_xlabel('일', fontsize=10)
    ax.set_ylabel('총 항공편 수', fontsize=10, color='blue')
    ax2.set_ylabel('평균 지연시간(분)', fontsize=10, color='red')
    ax3.set_ylabel('지연 비율(%)', fontsize=10, color='orange')
    ax.set_title(f'{month_names[i]}', fontsize=12, fontweight='bold')
    
    # 눈금 색상 설정
    ax.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
    ax3.tick_params(axis='y', labelcolor='orange', labelsize=8)
    
    # 격자 표시
    ax.grid(True, alpha=0.3)
    
    # x축 범위 설정
    ax.set_xlim(0.5, len(month_data['day']) + 0.5)

# 범례 추가 (첫 번째 서브플롯에만)
ax_legend = axes[0,0]
ax2_legend = axes[0,0].twinx()
ax3_legend = axes[0,0].twinx()

lines1, labels1 = ax_legend.get_legend_handles_labels()
lines2, labels2 = ax2_legend.get_legend_handles_labels()
lines3, labels3 = ax3_legend.get_legend_handles_labels()

ax_legend.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.show()
#########################################################################################################################

# 항공사 이름 매핑
airline_names = {
    'UA': 'United Airlines',
    'B6': 'JetBlue Airways', 
    'EV': 'ExpressJet'
}

# 각 항공사별 데이터 저장
airlines_data = {
    'UA': UA_total,
    'B6': B6_total,
    'EV': EV_total
}

# 각 항공사별 월별/일별 분석 및 시각화
def create_airline_monthly_daily_chart(carrier_data, carrier_code):
    """항공사별 월별/일별 지연 분석 차트 생성"""
    
    # 15분 이상 지연된 항공편만 필터링하여 평균 계산
    delayed_flights = carrier_data[carrier_data['dep_delay'] >= 15]
    
    # 월별/일별 데이터 그룹화
    monthly_data = carrier_data.groupby(['month','day']).agg({
        'year': 'count',  # 총 항공편 수
    }).reset_index()
    
    # 15분 이상 지연된 항공편의 평균 지연시간 계산
    delayed_avg = delayed_flights.groupby(['month','day'])['dep_delay'].mean().reset_index()
    delayed_avg.columns = ['month', 'day', 'avg_delay_15plus']
    
    # 15분 이상 지연된 항공편 수 계산
    delayed_count = delayed_flights.groupby(['month','day']).size().reset_index(name='delayed_count')
    
    # 데이터 병합
    monthly_data = monthly_data.merge(delayed_avg, on=['month', 'day'], how='left')
    monthly_data = monthly_data.merge(delayed_count, on=['month', 'day'], how='left')
    
    # NaN 값을 0으로 채우기 (지연이 없는 날)
    monthly_data['avg_delay_15plus'] = monthly_data['avg_delay_15plus'].fillna(0)
    monthly_data['delayed_count'] = monthly_data['delayed_count'].fillna(0)
    
    # 지연 비율 계산 (15분 이상 지연 / 전체 항공편)
    monthly_data['delay_ratio'] = (monthly_data['delayed_count'] / monthly_data['year']) * 100
    
    # 컬럼명 변경
    monthly_data.columns = ['month', 'day', 'total_flights', 'avg_delay_15plus', 'delayed_count', 'delay_ratio']
    
    # 12개 월별 서브플롯 생성
    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle(f'{airline_names[carrier_code]} - 월별 항공편 현황 (지연 기준: 15분 이상)', 
                 fontsize=18, fontweight='bold')
    
    # 월 이름
    month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                   '7월', '8월', '9월', '10월', '11월', '12월']
    
    for i in range(12):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 해당 월 데이터 필터링
        month_data = monthly_data[monthly_data['month'] == i+1]
        
        if month_data.empty:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{month_names[i]}', fontsize=12, fontweight='bold')
            continue
        
        # 삼중 y축 생성
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        
        # ax3의 spine을 오른쪽으로 이동
        ax3.spines['right'].set_position(('outward', 60))
        
        # 막대 그래프 (총 항공편 수)
        bars = ax.bar(month_data['day'], month_data['total_flights'], 
                      alpha=0.6, color='lightblue', label='총 항공편 수', width=0.8)
        
        # 선 그래프 (15분 이상 지연의 평균 지연시간)
        # 지연이 있는 날만 표시 (0이 아닌 값만)
        delay_data = month_data[month_data['avg_delay_15plus'] > 0]
        if not delay_data.empty:
            line1 = ax2.plot(delay_data['day'], delay_data['avg_delay_15plus'], 
                            color='red', marker='o', markersize=4, linewidth=2, 
                            label='평균 지연시간(15분+)')
        
        # 선 그래프 (지연 비율)
        line2 = ax3.plot(month_data['day'], month_data['delay_ratio'], 
                        color='orange', marker='s', markersize=3, linewidth=2, 
                        label='지연 비율(%)')
        
        # 축 설정
        ax.set_xlabel('일', fontsize=10)
        ax.set_ylabel('총 항공편 수', fontsize=10, color='blue')
        ax2.set_ylabel('평균 지연시간(분)', fontsize=10, color='red')
        ax3.set_ylabel('지연 비율(%)', fontsize=10, color='orange')
        ax.set_title(f'{month_names[i]}', fontsize=12, fontweight='bold')
        
        # 눈금 색상 설정
        ax.tick_params(axis='y', labelcolor='blue', labelsize=8)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
        ax3.tick_params(axis='y', labelcolor='orange', labelsize=8)
        
        # 격자 표시
        ax.grid(True, alpha=0.3)
        
        # x축 범위 설정
        if len(month_data['day']) > 0:
            ax.set_xlim(0.5, max(month_data['day']) + 0.5)
    
    # 범례 추가 (첫 번째 서브플롯에만)
    ax_legend = axes[0,0]
    ax2_legend = axes[0,0].twinx()
    ax3_legend = axes[0,0].twinx()
    
    lines1, labels1 = ax_legend.get_legend_handles_labels()
    lines2, labels2 = ax2_legend.get_legend_handles_labels()
    lines3, labels3 = ax3_legend.get_legend_handles_labels()
    
    ax_legend.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                    loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.show()

# 각 항공사별로 차트 생성
for carrier_code, carrier_data in airlines_data.items():
    print(f"\n{'='*60}")
    print(f"{airline_names[carrier_code]} 분석 시작...")
    print(f"총 데이터 수: {len(carrier_data):,}개")
    print(f"{'='*60}")
    
    if not carrier_data.empty:
        create_airline_monthly_daily_chart(carrier_data, carrier_code)
    else:
        print(f"{airline_names[carrier_code]}의 데이터가 없습니다.")

print(f"\n{'='*60}")
print("모든 항공사 분석 완료!")
print(f"{'='*60}")

merged_total = pd.merge(flights_cleaned,df_planes,on='tailnum',how='left')
merged_EV = merged_total[merged_total['carrier']=='EV'].reset_index()
merged_EV.shape
a = merged_EV['tailnum'].value_counts()
type(a)
