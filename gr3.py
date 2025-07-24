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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터
labels = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
EV_counts = [4300, 4150, 4500, 4600, 4550, 4600, 4700, 4750, 4540, 4810, 4600, 4380]
B6_counts = [4400, 4200, 4700, 4650, 4580, 4550, 4900, 4940, 4700, 4700, 4630, 4650]
UA_counts = [4500, 4350, 4600, 4610, 4590, 4550, 4920, 4900, 4800, 4750, 4700, 4390]
Total_counts = [27500, 27300, 28000, 27850, 27900, 27600, 28100, 28300, 27900, 27800, 27700, 27400]

# 색상 설정
color_dict = {
    'EV': '#d62728',
    'B6': '#2ca02c',
    'UA': '#1f77b4',
    'Total': '#7f7f7f'
}

# Plot 설정
fig, (ax_upper, ax_lower) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                         gridspec_kw={'height_ratios': [1, 3]})
fig.suptitle('항공사별 월별 운항 횟수', fontsize=20, fontweight='bold')

x = range(len(labels))

# 선 그래프 그리기
ax_lower.plot(x, EV_counts, label='EV', marker='o', color=color_dict['EV'], linewidth=4)
ax_lower.plot(x, B6_counts, label='B6', marker='o', color=color_dict['B6'])
ax_lower.plot(x, UA_counts, label='UA', marker='o', color=color_dict['UA'])
ax_upper.plot(x, Total_counts, label='Total', marker='o', color=color_dict['Total'])

# 점 위/아래 텍스트 표시
for i in x:
    ax_lower.text(i, EV_counts[i] - 70, str(EV_counts[i]), ha='center', va='top', fontsize=9, color=color_dict['EV'])     # EV 아래
    ax_lower.text(i, B6_counts[i] + 50, str(B6_counts[i]), ha='center', va='bottom', fontsize=9, color=color_dict['B6'])  # B6 위
    ax_lower.text(i, UA_counts[i] - 70, str(UA_counts[i]), ha='center', va='top', fontsize=9, color=color_dict['UA'])     # UA 아래
    ax_upper.text(i, Total_counts[i] + 150, str(Total_counts[i]), ha='center', va='bottom', fontsize=9, color=color_dict['Total'])  # Total 위

# Y축 범위
ax_lower.set_ylim(4000, 5100)
ax_upper.set_ylim(27000, 28500)

# 물결선 표시
d = .01
kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
ax_upper.plot((-d, +d), (-d, +d), **kwargs)
ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_lower.transAxes)
ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# x축 설정
ax_lower.set_xticks(x)
ax_lower.set_xticklabels(labels, fontsize=12)
ax_lower.set_xlabel('월', fontsize=12)
ax_lower.set_ylabel('운항 횟수', fontsize=12)

# 범례
custom_lines = [
    Line2D([0], [0], color=color_dict['UA'], linestyle='-', marker='o', linewidth=2, label='UA'),
    Line2D([0], [0], color=color_dict['B6'], linestyle='-', marker='o', linewidth=2, label='B6'),
    Line2D([0], [0], color=color_dict['EV'], linestyle='-', marker='o', linewidth=2, label='EV')
]
ax_upper.legend(
    handles=custom_lines,
    title='항공사',
    fontsize=10,
    title_fontsize=12,
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0)
)

plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.show()




################################################################
EV_total = flights_cleaned[flights_cleaned['carrier']=='EV']
EV_flight=EV_total[['carrier','distance','air_time']]

# 비율(%) 및 라벨 처리
for df in [UA_total_group, B6_total_group, EV_total_group, total_group]:
    df['delay_ratio_percent'] = df['delay_ratio'] * 100
    df['month_str'] = df['month'].astype(str) + '월'

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# 색상 정의
color_dict = {
    'EV': '#d62728',     # 빨강
    'B6': '#2ca02c',     # 초록
    'UA': '#1f77b4',     # 파랑
    'Total': '#7f7f7f'   # 회색
}

# 월 문자열 추가
for df in [EV_total_group, B6_total_group, UA_total_group, total_group]:
    df['month_str'] = df['month'].astype(str) + '월'

# 그래프 그리기
plt.figure(figsize=(14, 8))
plt.title('항공사별 월별 지연 비율 (%)', fontsize=20, fontweight='bold')

# 선 그래프 그리기
for label, df in {
    'EV': EV_total_group,
    'B6': B6_total_group,
    'UA': UA_total_group,
    'Total': total_group
}.items():
    plt.plot(df['month_str'], df['delay_ratio_percent'], label=label,
             marker='o',
             linewidth=4 if label == 'EV' else 2,   # EV만 굵게!
             color=color_dict[label])

    # 텍스트 위치 및 표시 조건 설정
    for x, y in zip(df['month_str'], df['delay_ratio_percent']):
        if label == 'UA':
            plt.text(x, y - 2, f"{y:.1f}%", ha='center', va='top', fontsize=15, color=color_dict[label])
        elif label == 'EV':
            plt.text(x, y + 1.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=15, 
                     color=color_dict[label], fontweight='bold')  
        else:
            plt.text(x, y + 1.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=15, color=color_dict[label])

# y축 퍼센트 포맷
plt.ylim(0, 70)
plt.ylabel('지연 비율 (%)', fontsize=12)
plt.xlabel('월', fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# 범례
plt.legend(title='항공사', fontsize=10, title_fontsize=12)

plt.tight_layout()
plt.show()




##################################################
#날씨에서 데이터 프레임

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

pivot_table