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

df_delayed = df_flights[df_flights['dep_delay'] >= 15]
df_delayed
df_planes
df_boeing = df_planes[df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)]
df_boeing
boeing_tailnums = df_boeing['tailnum'].dropna().unique()
print(list(boeing_tailnums))

boeing_tailnums = df_planes[
    df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)
]['tailnum'].dropna().unique()

df_boeing_delayed = df_flights[
    (df_flights['dep_delay'] >= 15) & 
    (df_flights['tailnum'].isin(boeing_tailnums))
]

print(df_boeing_delayed.head())
print(f"보잉 항공기 중 15분 이상 지연된 항공편 수: {len(df_boeing_delayed)}")
#보잉 항공기의 tailnum
boeing_tailnums = df_planes[
    df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)
]['tailnum'].dropna().unique()

# 출발 지연 15분 이상 ,보잉 항공기
df_boeing_delayed = df_flights[
    (df_flights['dep_delay'] >= 15) &
    (df_flights['tailnum'].isin(boeing_tailnums))
]
 
df_boeing_delayed

# 보잉 항공기의 tailnum 목록
boeing_tailnums = df_planes[
    df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)
]['tailnum'].dropna().unique()

# 지연 15분 이상,보잉 항공기의 tailnum
delayed_boeing_tailnums = df_flights[
    (df_flights['dep_delay'] >= 15) &
    (df_flights['tailnum'].isin(boeing_tailnums))
]['tailnum'].dropna().unique()

print(delayed_boeing_tailnums)
print(list(delayed_boeing_tailnums))

# tailnum별 지연 표?
tailnum_delay_counts = df_boeing_delayed['tailnum'].value_counts().reset_index()
tailnum_delay_counts.columns = ['tailnum', 'delay_count']

print(tailnum_delay_counts)

#비교쇼
df_planes[df_planes['tailnum'] == 'N327AA']
df_planes[df_planes['tailnum'] == 'N335AA']
df_planes[df_planes['tailnum'] == 'N332AA']
df_planes[df_planes['tailnum'] == 'N336AA']
df_planes[df_planes['tailnum'] == 'N319AA']
df_planes[df_planes['tailnum'] == 'N654SW']
df_planes[df_planes['tailnum'] == 'N524AS']
df_planes[df_planes['tailnum'] == 'N370SW']

#지연된 보잉기 갖고있는 항공사 수
df_boeing_delayed['carrier'].value_counts()

df_boeing_all = df_flights[df_flights['tailnum'].isin(boeing_tailnums)]
delay_rate_by_carrier = (
    df_boeing_delayed['carrier'].value_counts() / df_boeing_all['carrier'].value_counts()
).dropna().sort_values(ascending=False)

df_joined = pd.merge(df_boeing_delayed, df_planes[['tailnum', 'model']], on='tailnum', how='left')
model_counts = df_joined['model'].value_counts()

df_joined.groupby('model')['dep_delay'].mean().sort_values(ascending=False)

boeing_old_tailnums = df_planes[
    (df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)) &
    (df_planes['year'].isin([1986, 1987]))
]['tailnum'].dropna().unique()

df_boeing_old_delay = df_flights[
    (df_flights['dep_delay'] >= 15) &
    (df_flights['tailnum'].isin(boeing_old_tailnums))
]

tailnum_delay_count = (
    df_boeing_old_delay['tailnum']
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'tailnum', 'tailnum': 'delay_count'})
    .sort_values(by='delay_count', ascending=True)
)

print(tailnum_delay_count.head(10))
df_planes[df_planes['tailnum'] == 'N121DE']
df_planes[df_planes['tailnum'] == 'N245AY']
df_planes[df_planes['tailnum'] == 'N327AA']
df_planes[df_planes['tailnum'] == 'N335AA']
#여기서 지연된 원인은 알수 없다
# 비슷한 년도에 지어진 것들을 tailnum뺴고 다 동일한데

boeing_tailnums = df_planes[
    df_planes['manufacturer'].str.contains('BOEING', case=False, na=False)
]['tailnum'].dropna().unique()
df_flights['arr_delay']
df_flights['arr_delay'].isna().sum()
df_clean = df_flights.dropna(subset=['arr_delay', 'dep_delay', 'tailnum'])
df_cancelled = df_flights[df_flights['cancelled'] == 1]
# 출발 시간이 기록되지 않은 항공편 = 결항
df_cancelled = df_flights[df_flights['dep_time'].isna()]

# 정상적으로 출발한 항공편
df_operated = df_flights[df_flights['dep_time'].notna()]

boeing_planes = df_planes[df_planes['manufacturer'].str.contains('BOEING', na=False)]
nan_dep_delay = df_flights[df_flights['dep_delay'].isna()]

nan_boeing_dep_delay = pd.merge(nan_dep_delay, boeing_planes, on='tailnum')

# 결과 확인
print(nan_boeing_dep_delay[['flight', 'tailnum', 'manufacturer', 'model', 'dep_delay']].head())
print("총 보잉사 NaN 출발지연 항공편 수:", len(nan_boeing_dep_delay))
print(df_flights['dep_delay'].unique())
print(df_flights['dep_delay'].value_counts(dropna=False).sort_index())

# 결항편 중 보잉 항공기만
cancelled_boeing = pd.merge(df_cancelled, boeing_planes, on='tailnum')

# tailnum 별 결항 횟수
cancelled_count_by_tailnum = cancelled_boeing['tailnum'].value_counts().reset_index()
cancelled_count_by_tailnum.columns = ['tailnum', 'cancelled_count']

# 결과 확인
print(cancelled_count_by_tailnum.head(10))

df_planes[df_planes['tailnum'] == 'N906AT']
df_planes[df_planes['tailnum'] == 'N678DL']
df_planes[df_planes['tailnum'] == 'N328AA']

#지연과 결항 동시에
boeing_planes = df_planes[df_planes['manufacturer'].str.contains('BOEING', na=False)]
delayed = df_flights[df_flights['dep_delay'] > 0]
cancelled = df_flights[df_flights['dep_time'].isna()]
boeing_tailnums = set(boeing_planes['tailnum'].dropna())
delayed_boeing_tailnums = set(delayed['tailnum'].dropna()) & boeing_tailnums
cancelled_boeing_tailnums = set(cancelled['tailnum'].dropna()) & boeing_tailnums

# 5. 지연,결항도 있었던 보잉기
both_boeing_tailnums = delayed_boeing_tailnums & cancelled_boeing_tailnums
len(both_boeing_tailnums)
df_both_boeing = df_flights[df_flights['tailnum'].isin(both_boeing_tailnums)]

#지연과 결항 합쳐서 많은 순서대로
boeing_delayed = df_flights[
    (df_flights['dep_delay'] >= 15) &
    (df_flights['tailnum'].isin(boeing_planes['tailnum']))
]

boeing_cancelled = df_flights[
    df_flights['dep_time'].isna() &
    (df_flights['tailnum'].isin(boeing_planes['tailnum']))
]
delay_count = boeing_delayed['tailnum'].value_counts().rename('delay_count')
cancel_count = boeing_cancelled['tailnum'].value_counts().rename('cancel_count')

combined = pd.concat([delay_count, cancel_count], axis=1).fillna(0)
combined['total'] = combined['delay_count'] + combined['cancel_count']
combined = combined.sort_values(by='total', ascending=False)
print(combined.head(10))

df_planes[df_planes['tailnum'] == 'N327AA']
df_planes[df_planes['tailnum'] == 'N332AA']
df_planes[df_planes['tailnum'] == 'N335AA']
df_planes[df_planes['tailnum'] == 'N319AA']


보잉사 ewr에서 거의 출발 50퍼센트 이상

model_counts = df_joined['model'].value_counts()

boeing_planes = df_planes[df_planes['manufacturer'].str.contains('BOEING', na=False)]

# 보잉사 항공기만 포함된 운항 데이터 추출
boeing_flights = pd.merge(df_flights, boeing_planes[['tailnum']], on='tailnum')

# 보잉사 tailnum별 운항 횟수 계산
boeing_tailnum_counts = boeing_flights['tailnum'].value_counts()

# 가장 많이 운항된 보잉사 tailnum 상위 10개 출력
print(boeing_tailnum_counts.head(10))
df_planes[df_planes['tailnum'] == 'N328AA']

# 해당 tailnum만 추출
n328aa_flights = df_flights[df_flights['tailnum'] == 'N328AA']

# 15분 이상 지연된 경우
n328aa_delayed = n328aa_flights[n328aa_flights['dep_delay'] >= 15]
print("15분 이상 지연된 건수:", len(n328aa_delayed))
display(n328aa_delayed[['year', 'month', 'day', 'dep_time', 'dep_delay', 'flight', 'origin', 'dest']])

# JFK에서 출발한 항공편만 필터링
jfk_flights = df_flights[df_flights['origin'] == 'JFK']
dest_counts = jfk_flights['dest'].value_counts().reset_index()
dest_counts.columns = ['dest', 'count']
print(dest_counts)

ewr_flights = df_flights[df_flights['origin'] == 'EWR']
dest_counts = ewr_flights['dest'].value_counts().reset_index()
dest_counts.columns = ['dest', 'count']
print(dest_counts)

lga_flights = df_flights[df_flights['origin'] == 'LGA']
dest_counts = lga_flights['dest'].value_counts().reset_index()
dest_counts.columns = ['dest', 'count']
print(dest_counts)

df_airlines[['carrier', 'name']]