import pandas as pd
import nycflights13 as flights
import matplotlib.pyplot as plt
import seaborn as sns
# 데이터프레임 할당
# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
flights_cleaned

# 3사의 지연율과 평균
# 분석 대상 항공사
selected_carriers = ['UA', 'B6', 'EV']
flights_cleaned = flights_cleaned[flights_cleaned['carrier'].isin(selected_carriers)]


# 지연율 및 평균 계산
summary = flights_cleaned.groupby('carrier').apply(
    lambda g: pd.Series({
        'total_flights': len(g),
        'delayed_flights': (g['dep_delay'] >= 15).sum(),
        'delay_rate (%)': round((g['dep_delay'] >= 15).mean() * 100, 2),
        'avg_delay (min)': round(g[g['dep_delay'] >= 15]['dep_delay'].mean(), 2)
    })
).reset_index()
summary = summary.sort_values("total_flights", ascending=False)
summary



# 3사의 공항에 따른 지연율
# 분석 대상 항공사
selected_carriers = ['UA', 'B6', 'EV']

# 15분 이상 지연된 항공편만 필터링
filtered = flights_cleaned[
    (flights_cleaned['carrier'].isin(selected_carriers)) &
    (flights_cleaned['dep_delay'] >= 15)
]

# 결측값 제거 후 평균 출발 지연 시간 계산 (carrier, origin 기준)
airport_delay = (
    filtered.dropna(subset=['dep_delay'])
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


#3항공사의 기상 상황에 따른 지연율
# 타겟 항공사 필터링
target_carriers = ['UA', 'B6', 'EV']
flights_cleaned = flights_cleaned[flights_cleaned['carrier'].isin(target_carriers)]

# 지연 여부 컬럼 추가
flights_cleaned['is_delayed'] = flights_cleaned['dep_delay'] >= 15

# 필요한 열만 추출
df_flights_subset = flights_cleaned[['year', 'month', 'day', 'hour', 'origin', 'carrier', 'is_delayed']]
df_weather_subset = df_weather[['year', 'month', 'day', 'hour', 'origin', 'precip', 'visib', 'temp', 'wind_speed']]

# 병합
merged = pd.merge(df_flights_subset, df_weather_subset, 
                  on=['year', 'month', 'day', 'hour', 'origin'], how='inner')
merged = merged.dropna()

# 시각화를 위한 구간 설정
merged['precip_bin'] = pd.cut(merged['precip'], bins=[-0.01, 0.01, 0.1, 0.5, 2], 
                              labels=['0', '0.01–0.1', '0.1–0.5', '0.5–2'])
merged['visib_bin'] = pd.cut(merged['visib'], bins=[0, 2, 5, 10], labels=['0–2', '2–5', '5–10'])
merged['wind_bin'] = pd.cut(merged['wind_speed'], bins=[0, 5, 10, 20, 40], labels=['0–5', '5–10', '10–20', '20+'])

# 원하는 순서 지정
carrier_order = ['UA', 'B6', 'EV']
merged['carrier'] = pd.Categorical(merged['carrier'], categories=carrier_order, ordered=True)

# 시각화용 함수
def plot_weather_factor(factor_col, label):
    group = merged.groupby(['carrier', factor_col])['is_delayed'].mean().reset_index()
    group['지연율(%)'] = group['is_delayed'] * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(data=group, x=factor_col, y='지연율(%)', hue='carrier', palette='Set2')
    plt.title(f"{label}에 따른 항공편 지연율 (UA, B6, EV)")
    plt.xlabel(label)
    plt.ylabel("지연율 (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 강수량, 시정, 풍속에 따른 지연 시각화
plot_weather_factor('precip_bin', '강수량 (inches)')
plot_weather_factor('visib_bin', '시정 (miles)')
plot_weather_factor('wind_bin', '풍속 (mph)')

#3항공사별 지연율과 전체 평균 지연율의 비교
import pandas as pd
import matplotlib.pyplot as plt

# 지연 기준: 출발 지연이 15분 이상
flights_cleaned["is_delayed"] = flights_cleaned["dep_delay"] >= 15

# 분석 대상 3개 항공사
carriers = ['UA', 'B6', 'EV']
df_3carriers = flights_cleaned[flights_cleaned['carrier'].isin(carriers)]

# 항공사별 지연율 계산
delay_rate = df_3carriers.groupby('carrier')['is_delayed'].mean().reset_index()
delay_rate.columns = ['carrier', 'delay_rate']

# 지정한 순서대로 정렬
delay_rate = delay_rate.set_index('carrier').reindex(carriers).reset_index()

# 전체 평균 지연율
overall_delay_rate = flights_cleaned['is_delayed'].mean()

# 시각화
plt.figure(figsize=(8, 5))
bars = plt.bar(delay_rate['carrier'], delay_rate['delay_rate'], color='skyblue')

# 전체 평균 지연율 점선
plt.axhline(overall_delay_rate, color='red', linestyle='--', label='전체 평균 지연율')

# 막대 위 수치
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom')

# 전체 평균 지연율 수치
plt.text(len(carriers) - 0.5, overall_delay_rate + 0.005,
         f'{overall_delay_rate:.1%}', color='red', ha='right', va='bottom')

plt.title('3개 항공사의 지연율 (15분 이상 기준)')
plt.ylabel('지연율')
plt.xlabel('항공사')
plt.ylim(0, max(delay_rate['delay_rate'].max(), overall_delay_rate) + 0.1)
plt.legend()
plt.tight_layout()
plt.show()



#EV 지연율과 날씨와의 관계
#######################################################
# 배우지 않은 부분인 것 같은데 강수량과 관련이 많이 있다는 내용의 그래프
# Y축은 딜레이가 됬다 안됬다 여부 판단
# 우상향 기울기가 높을수록 관련이 높다고 볼 수 있음.

# EV 항공편 필터링
df_ev = flights_cleaned[flights_cleaned['carrier'] == 'EV'].copy()

# 날짜 기반 키 생성
df_ev['time_hour'] = pd.to_datetime(df_ev['time_hour'])
df_weather['time_hour'] = pd.to_datetime(df_weather['time_hour'])

# 지연 여부 컬럼 추가 (15분 이상 지연)
df_ev['is_delayed'] = df_ev['dep_delay'] >= 15

# 날씨 데이터와 병합 (origin + time_hour 기준)
df_ev_weather = pd.merge(
    df_ev,
    df_weather,
    on=['origin', 'time_hour'],
    how='left'
)

# 필요한 컬럼만 선택
columns_of_interest = ['is_delayed', 'temp', 'wind_speed', 'visib', 'precip']
df_ev_weather = df_ev_weather[columns_of_interest].dropna()

# 상관계수 출력
correlation = df_ev_weather.corr(numeric_only=True)
print(correlation['is_delayed'].sort_values(ascending=False))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

weather_vars = ['temp', 'wind_speed', 'visib', 'precip']
for i, col in enumerate(weather_vars, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=col, y='is_delayed', data=df_ev_weather, alpha=0.3)
    sns.regplot(x=col, y='is_delayed', data=df_ev_weather, scatter=False, color='red')
    plt.title(f"{col} vs. 지연율")
    plt.ylabel("지연율")
    plt.xlabel(col)

plt.tight_layout()
plt.show()