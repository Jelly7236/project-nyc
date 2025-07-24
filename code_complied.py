import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nycflights13 as flights
import matplotlib.font_manager as fm

font_path = r"C:\Windows\Fonts\malgun.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

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

##################################################################################
######################################################################################
## 전체 항공사 별 지연율 비교 시각화
####################################################################################
#############################################################

bar_colors = ['#d62728' if c == 'EV' else '#1f77b4' for c in summary_big3['carrier']]

plt.figure(figsize=(8,6))
sns.barplot(data=summary_big3, x='carrier', y='delay_rate (%)', palette=bar_colors)

plt.title('항공사별 출발 지연 비율(%)', fontsize=18, fontweight='bold')   # 제목 크기 키움 + 볼드체
plt.xlabel('항공사', fontsize=14, fontweight='bold')              # x축 라벨 크기 키움 + 볼드체
plt.ylabel('지연 비율 (%)', fontsize=14, fontweight='bold')       # y축 라벨 크기 키움 + 볼드체

# x축 항공사 이름 (눈금 레이블) 볼드체, 크기 14
plt.xticks(fontsize=14, fontweight='bold')

for i, rate in enumerate(summary_big3['delay_rate (%)']):
    plt.text(i, rate + 0.5, f'{rate}%', ha='center', va='bottom', fontsize=13, fontweight='bold')  # 텍스트 크기 키움 + 볼드체

plt.ylim(0, summary_big3['delay_rate (%)'].max() + 5)
plt.tight_layout()
plt.show()
######################################################################################
## 각 공항별 지연시간 평균 시각화
#####################################################################
# 전체 항공편 수 (공항별)
total_flights = flights_cleaned.groupby('origin').size().reset_index(name='total_flights')

# 15분 이상 지연된 항공편 수 (공항별)
delayed_flights = flights_cleaned[flights_cleaned['dep_delay'] >= 15]
delayed_counts = delayed_flights.groupby('origin').size().reset_index(name='delayed_flights')

# 비율 계산
delay_ratio = pd.merge(total_flights, delayed_counts, on='origin', how='left')
delay_ratio['delayed_flights'] = delay_ratio['delayed_flights'].fillna(0)
delay_ratio['delay_rate'] = delay_ratio['delayed_flights'] / delay_ratio['total_flights']
delay_ratio = delay_ratio.sort_values('delay_rate', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(data=delay_ratio, x='origin', y='delay_rate', color='#1f77b4')

# 타이틀 및 축 설정
plt.title('공항별 출발 지연 비율 (15분 이상 지연 기준)', fontsize=18, fontweight='bold')
plt.xlabel('출발 공항 코드', fontsize=14, fontweight='bold')
plt.ylabel('지연 비율 (%)', fontsize=14, fontweight='bold')

# 막대 위에 수치 표시 (퍼센트 형식)
for i, value in enumerate(delay_ratio['delay_rate']):
    plt.text(i, value + 0.002, f'{value*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 폰트 및 범위 조정
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)
plt.ylim(0, delay_ratio['delay_rate'].max() + 0.02)
plt.tight_layout()
plt.show()

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

###################################################################################
###################################################################################
#######################################################################################
# 월별 운항 횟수 시각화
########################################################################
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
# 항공사별 월별 지연 비율 시각화
############################################################
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
##################################################################################
# 시각화
##################################################################################
# 3사 비행기의 제조사 비율 시각화

# 제조사별 색상 고정
manufacturer_colors = {
    'BOEING': '#1f77b4',
    'AIRBUS': '#ff7f0e',
    'MCDONNELL DOUGLAS': '#2ca02c',
    'EMBRAER': '#d62728',
    'BOMBARDIER': '#9467bd',
    'CANADAIR': '#8c564b',
    'OTHER': '#aaaaaa'
}

carriers = plane_features['carrier'].unique()

for carrier in carriers:
    df_sub = plane_features[plane_features['carrier'] == carrier].copy()
    df_sub = df_sub.sort_values(by='count', ascending=False)

    total_planes = df_sub['count'].sum()

    # 비율 계산
    df_sub['ratio'] = df_sub['count'] / total_planes

    # 2% 미만 제조사는 기타로 묶기
    major = df_sub[df_sub['ratio'] >= 0.02].copy()
    minor = df_sub[df_sub['ratio'] < 0.02].copy()

    if not minor.empty:
        other_row = pd.DataFrame({
            'manufacturer': ['OTHER'],
            'count': [minor['count'].sum()],
            'ratio': [minor['count'].sum() / total_planes]
        })
        major = pd.concat([major[['manufacturer', 'count', 'ratio']], other_row], ignore_index=True)
    else:
        major = df_sub[['manufacturer', 'count', 'ratio']]

    major = major.reset_index(drop=True)
    labels = major['manufacturer']
    sizes = major['count']
    colors = [manufacturer_colors.get(mfg, '#cccccc') for mfg in labels]

    # 비율 계산 (누적 각도 계산용)
    ratios = sizes / sizes.sum()
    angles = ratios * 360
    cum_sizes = angles.cumsum()
    start_angles = np.concatenate(([0], cum_sizes[:-1]))

    # 가장 큰 wedge(1등)의 인덱스
    max_idx = major['count'].idxmax()

    # 2, 3등 중 하나의 중심각을 60도(1시 방향)에 위치하도록 회전각 계산
    top_idx = major['count'].nlargest(3).index.tolist()
    top_idx.remove(max_idx)
    second_idx = top_idx[0]

    start_angle_2nd = start_angles[second_idx]
    end_angle_2nd = cum_sizes[second_idx]
    mid_angle_2nd = (start_angle_2nd + end_angle_2nd) / 2

    # 2등 중심을 60도로 맞추도록 회전
    startangle = 60 - mid_angle_2nd

    # 파이차트 그리기
    plt.figure(figsize=(7, 7))
    wedges, _ = plt.pie(
        sizes,
        labels=None,
        startangle=startangle,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    )

    # 상위 2개 제조사 인덱스
    top2_idx = major['count'].nlargest(2).index.tolist()

    for i, w in enumerate(wedges):
        if i in top2_idx:
            angle = (w.theta2 + w.theta1) / 2
            angle_rad = np.deg2rad(angle)
            r = 0.55
            x = r * np.cos(angle_rad)
            y = r * np.sin(angle_rad)

            ratio_text = f"{(sizes.iloc[i] / total_planes) * 100:.1f}% ({sizes.iloc[i]}대)"
            plt.text(x, y + 0.05, ratio_text, ha='center', va='center',
                     fontsize=13, fontweight='bold', color='black')
            plt.text(x, y - 0.05, labels.iloc[i], ha='center', va='center',
                     fontsize=12, color='black')

    plt.title(f'{carrier} 항공사의 제조사 비율', fontsize=16, pad=30)
    plt.text(1.1, -1.2, f'총 비행기 수: {total_planes}', fontsize=12, ha='right')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()
######################################################################################################
# 각 제조사별 평균 좌석수 시각화

# 주요 제조사
major_manufacturers = ['BOEING', 'EMBRAER', 'AIRBUS', 'MCDONNELL DOUGLAS']

# 평균 좌석 수 계산
if 'avg_seats' not in plane_features.columns:
    avg_seats_by_mfg = (
        big3.dropna(subset=['manufacturer', 'seats'])
        .groupby('manufacturer')['seats']
        .mean()
        .reset_index()
    )
else:
    avg_seats_by_mfg = plane_features.groupby('manufacturer')['avg_seats'].mean().reset_index()

# 주요 제조사만 필터링
avg_seats_by_mfg = avg_seats_by_mfg[avg_seats_by_mfg['manufacturer'].isin(major_manufacturers)]

# 색상 설정
colors = [manufacturer_colors.get(mfg, '#cccccc') for mfg in avg_seats_by_mfg['manufacturer']]

# 시각화
plt.figure(figsize=(8, 6))
bars = plt.bar(
    avg_seats_by_mfg['manufacturer'],
    avg_seats_by_mfg['seats'] if 'seats' in avg_seats_by_mfg.columns else avg_seats_by_mfg['avg_seats'],
    color=colors
)

# 타이틀 & 축 설정
plt.title('제조사별 평균 좌석 수 비교', fontsize=18, fontweight='bold')
plt.xlabel('제조사', fontsize=14, fontweight='bold')
plt.ylabel('평균 좌석 수', fontsize=14, fontweight='bold')

# 제조사 이름 글꼴 키우고 볼드 처리
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

# 막대 위에 수치 표시
for bar in bars:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    plt.text(x, height + 2, f'{height:.0f}개', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylim(0, max([bar.get_height() for bar in bars]) + 20)
plt.tight_layout()
plt.show()

##########################################################################################
# 각 항공사별 평균 좌석수 시각화
# 색상 고정
carrier_colors = {
    'EV': '#d62728',   # Red
    'UA': '#1f77b4',   # Blue
    'B6': '#1f77b4'    # Green
}

# 정렬
avg_seats_sorted = avg_seats.sort_values(by='avg_seats', ascending=False).reset_index(drop=True)

# 색상 리스트 생성
bar_colors = [carrier_colors.get(carrier, 'gray') for carrier in avg_seats_sorted['carrier']]

# 시각화
plt.figure(figsize=(9, 6))
bars = sns.barplot(data=avg_seats_sorted, x='carrier', y='avg_seats', palette=bar_colors)

# 수치 텍스트 박스 추가
for bar, value in zip(bars.patches, avg_seats_sorted['avg_seats']):
    bars.annotate(f"{value:.1f}",
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  ha='center', va='bottom',
                  fontsize=13, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

# 제목과 라벨
plt.title("항공사별 평균 좌석 수", fontsize=18, weight='bold')
plt.xlabel("항공사", fontsize=13)
plt.ylabel("평균 좌석 수", fontsize=13)
plt.ylim(0, avg_seats_sorted['avg_seats'].max() + 30)

# 범례 수동 추가
custom_labels = [plt.Rectangle((0,0),1,1, color=carrier_colors[c]) for c in avg_seats_sorted['carrier']]
plt.legend(custom_labels, avg_seats_sorted['carrier'], title="항공사", loc='upper right')

plt.tight_layout()
plt.show()

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

top_planes = schedule_summary.sort_values('avg_gap_hr').head(10)

top_planes_with_seats = pd.merge(
    top_planes,
    df_planes[['tailnum', 'seats']],
    on='tailnum',
    how='left'
)

# 색상: 상위 4개는 빨간색, 나머지는 파란색
colors = ['#d62728' if i < 4 else '#1f77b4' for i in range(len(top_planes_with_seats))]

plt.figure(figsize=(10, 5))
barplot = sns.barplot(data=top_planes_with_seats, x='tailnum', y='avg_gap_hr', palette=colors)

plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.title('평균 비행 간격이 가장 짧은 10대 EV 항공기', fontsize=16, fontweight='bold')
plt.xlabel('항공기 꼬리 번호', fontsize=14, fontweight='bold')
plt.ylabel('평균 비행 간격 (시간)', fontsize=14, fontweight='bold')
plt.grid(True)

# 막대 위에 좌석 수 표시 (한글, 볼드, 글자 크기 업)
for i, row in top_planes_with_seats.iterrows():
    if pd.notna(row['seats']):
        plt.text(
            x=i,
            y=row['avg_gap_hr'] + 0.2,
            s=f"{int(row['seats'])}석",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

plt.tight_layout()
plt.show()

################################################################################
# UA 기체 회전율
# UA 항공사의 기체 회전율 비교
ua_schedule = UA_total[['tailnum', 'month_day_time']].dropna()

# 1. datetime 형식으로 먼저 변환
ua_schedule['month_day_time'] = pd.to_datetime(ua_schedule['month_day_time'])

ua_schedule = ua_schedule.sort_values(['tailnum', 'month_day_time'])

# 2. tailnum 기준으로 시간 차이(diff) 계산 후, 이를 time_gap 이라는 새로운 column으로 추가

ua_schedule['time_gap'] = ua_schedule.groupby('tailnum')['month_day_time'].diff()

# 값이 NaT인 경우, 
# 앞쪽 값들 (NaT): 해당 tailnum 그룹에서 첫 비행 → 비교 대상 없음 → NaT

# 간격을 시간(hour) 단위로 변경
ua_schedule['gap_hours'] = ua_schedule['time_gap'].dt.total_seconds() / 3600

# 3. 결과 확인
ua_schedule['time_gap']
print(ua_schedule.head(10))

# 4. 값 확인
schedule_summary_ua = ua_schedule.groupby('tailnum')['gap_hours'].agg(['mean', 'min', 'count']).reset_index().sort_values('count',ascending=False)

schedule_summary_ua['count'].describe()

schedule_summary_ua.dropna()

schedule_summary_ua.columns = ['tailnum', 'avg_gap_hr', 'min_gap_hr', 'flight_count']

top_planes_ua = schedule_summary_ua.sort_values('avg_gap_hr').head(10)

# 좌석수 정보 병합
top_planes_ua_with_seats = pd.merge(
    top_planes_ua,
    df_planes[['tailnum', 'seats']],
    on='tailnum',
    how='left'
)

plt.figure(figsize=(10, 5))
barplot = sns.barplot(
    data=top_planes_ua_with_seats,
    x='tailnum',
    y='avg_gap_hr',
    color='#1f77b4'  # 파란색
)

plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.title('평균 비행 간격이 가장 짧은 10대 UA 항공기', fontsize=16, fontweight='bold')
plt.xlabel('항공기 꼬리 번호', fontsize=14, fontweight='bold')
plt.ylabel('평균 비행 간격 (시간)', fontsize=14, fontweight='bold')
plt.grid(True)

# 막대 위에 좌석 수 표시
for i, row in top_planes_ua_with_seats.iterrows():
    if pd.notna(row['seats']):
        plt.text(
            x=i,
            y=row['avg_gap_hr'] + 0.2,
            s=f"{int(row['seats'])}석",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

plt.tight_layout()
plt.show()
#################################################################################
# B6 기체 회전율
# B6 항공사 중에서 tailnum, 출발 시간, 날짜 정보 추출
b6_schedule = B6_total[['tailnum', 'month_day_time']].dropna()

# 1. datetime 형식으로 먼저 변환
b6_schedule['month_day_time'] = pd.to_datetime(b6_schedule['month_day_time'])

b6_schedule = b6_schedule.sort_values(['tailnum', 'month_day_time'])

# 2. tailnum 기준으로 시간 차이(diff) 계산 후, 이를 time_gap 이라는 새로운 column으로 추가
b6_schedule['time_gap'] = b6_schedule.groupby('tailnum')['month_day_time'].diff()

# 값이 NaT인 경우, 
# 앞쪽 값들 (NaT): 해당 tailnum 그룹에서 첫 비행 → 비교 대상 없음 → NaT

# 간격을 시간(hour) 단위로 변경
b6_schedule['gap_hours'] = b6_schedule['time_gap'].dt.total_seconds() / 3600

# 3. 결과 확인
b6_schedule['time_gap']
print(b6_schedule.head(10))

# 4. 값 확인
schedule_summary_b6 = b6_schedule.groupby('tailnum')['gap_hours'].agg(['mean', 'min', 'count']).reset_index().sort_values('count',ascending=False)

schedule_summary_b6['count'].describe()

schedule_summary_b6.dropna()

schedule_summary_b6.columns = ['tailnum', 'avg_gap_hr', 'min_gap_hr', 'flight_count']

top_planes_b6 = schedule_summary_b6.sort_values('avg_gap_hr').head(10)

# 좌석수 정보 병합
top_planes_b6_with_seats = pd.merge(
    top_planes_b6,
    df_planes[['tailnum', 'seats']],
    on='tailnum',
    how='left'
)

plt.figure(figsize=(10, 5))
barplot = sns.barplot(
    data=top_planes_b6_with_seats,
    x='tailnum',
    y='avg_gap_hr',
    color='#1f77b4'  # 파란색
)

plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.title('평균 비행 간격이 가장 짧은 10대 B6 항공기', fontsize=16, fontweight='bold')
plt.xlabel('항공기 꼬리 번호', fontsize=14, fontweight='bold')
plt.ylabel('평균 비행 간격 (시간)', fontsize=14, fontweight='bold')
plt.grid(True)

# 막대 위에 좌석 수 표시
for i, row in top_planes_b6_with_seats.iterrows():
    if pd.notna(row['seats']):
        plt.text(
            x=i,
            y=row['avg_gap_hr'] + 0.2,
            s=f"{int(row['seats'])}석",
            ha='center',
            va='bottom',
            fontsize=12,
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

## 공항별 평균 거리와 비행시간 시각화
plt.figure(figsize=(14, 5))

# 색상 매핑 함수
def get_color(airport):
    if airport == 'JFK':
        return '#2ca02c'  # 초록
    elif airport == 'LGA':
        return '#d62728'  # 빨강
    else:
        return '#1f77b4'  # 기본 파랑

# 1. 공항별 평균 거리 시각화
plt.subplot(1, 2, 1)
colors_dist = origin_summary['origin'].apply(get_color).tolist()
sns.barplot(data=origin_summary, x='origin', y='distance', palette=colors_dist)
plt.title('공항별 평균 거리 (마일)', fontsize=16, fontweight='bold')
plt.xlabel('출발 공항 코드', fontsize=14, fontweight='bold')
plt.ylabel('평균 거리 (마일)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

for i, val in enumerate(origin_summary['distance']):
    plt.text(i, val + 10, f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. 공항별 평균 비행 시간 시각화
plt.subplot(1, 2, 2)
colors_time = origin_summary['origin'].apply(get_color).tolist()
sns.barplot(data=origin_summary, x='origin', y='air_time', palette=colors_time)
plt.title('공항별 평균 비행 시간 (분)', fontsize=16, fontweight='bold')
plt.xlabel('출발 공항 코드', fontsize=14, fontweight='bold')
plt.ylabel('평균 비행 시간 (분)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

for i, val in enumerate(origin_summary['air_time']):
    plt.text(i, val + 2, f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

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


##############################################################################

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
######################################################################3
######################################################################
###############################################################################3
##################################################################################3
# 각 공항별 항공사 지연 시간 시각화
# carrier별 색상 딕셔너리 (EV 빨강, 나머지 파랑)
palette_dict = {
    'EV': '#d62728',
    'UA': '#1f77b4',
    'B6': '#1f77b4',
}

plt.figure(figsize=(12, 7))
barplot = sns.barplot(
    data=airport_delay,
    x='origin',
    y='dep_delay',   # 평균 지연 시간 컬럼명으로 맞춰주세요
    hue='carrier',
    palette=palette_dict
)

plt.title('공항별 항공사 평균 출발 지연 시간 (분)', fontsize=20, fontweight='bold')
plt.xlabel('출발 공항 코드', fontsize=16, fontweight='bold')
plt.ylabel('평균 지연 시간 (분)', fontsize=16, fontweight='bold')

# 막대 위에 지연 시간 표시 (0보다 클 때만)
for p in barplot.patches:
    height = p.get_height()
    if height > 0:
        plt.text(
            p.get_x() + p.get_width()/2,
            height + 0.5,
            f'{height:.1f}분',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

plt.legend(title='항공사', title_fontsize=14, fontsize=12)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
# 원하는 carrier 순서로 지정
carrier_order = ['UA', 'B6', 'EV']
airport_delay['carrier'] = pd.Categorical(airport_delay['carrier'], categories=carrier_order, ordered=True)

# 공항 이름 붙이기 (origin code -> 공항명)
airport_names = df_airports[['faa', 'name']].rename(columns={'faa': 'origin', 'name': 'airport_name'})
airport_delay = pd.merge(airport_delay, airport_names, on='origin', how='left')
