# conda env export > environment.yml
# conda env create -f environment.yml
import pandas as pd
import nycflights13 as flights
import numpy as np
# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# df_fights 구성 확인
df_flights.info()
df_flights.shape # (336776, 19)

# 'hour', 'minute'은 'sched_dep_time'을 나누어서 추출됨

# 결측치 처리
df_flights.isnull().sum()

# dep_time(실제 이륙 시간)과 dep_delay(이륙 지연 시간)의 null값 갯수가 같음 8255
# 해당 열 상태 확인
dep_null_idx = df_flights['dep_time'].isna() # dep_time이 null값인 데이터 프레임 생성
dep_null = df_flights.loc[dep_null_idx,:] # dep_null
dep_null.head(10)
# 상위 10개 값의 경우 출발 시간이 null값이면 도착 시간과 비행 시간 열도 null값임을 확인 후 검증
dep_null[['dep_time','dep_delay','arr_time','arr_delay','air_time']].isnull().sum()
# isnull().sum()을 한 결과 값이 모두 8255로 동일함 (전체 행의 갯수= 8255)

# 결과 1. dep_null에 속한 항공편은 이륙 자체를 하지 않음 = 예약된 항공편 아예 취소
# 해당 데이터 행을 완전히 배제하고 생각할지 
# 아니면 취소될 만큼 심각한 사안이 있었는지 알아보고 딜레이 시간에 미친 영향력 고려할지 선택해야함

# 우선 해당 인덱스 번호를 제외하고 데이터 프레임 생성 - df_flights2 
df_flights2 = df_flights.loc[~dep_null_idx,:].reset_index(drop=True)
df_flights.shape[0] - df_flights2.shape[0] # 8255로 올바르게 삭제 완료

# air_time과 arr_delay가 1175개로 동일한 갯수의 결측치를 갖고 있지만 
# 공항별 시차가 다르기 때문에 일일이 계산하기엔 무리가 있어서 남은 결측치도 삭제
df_flights2.isnull().sum()
df_flights2.dropna(inplace=True)
df_flights2.reset_index(inplace=True,drop=True)
df_flights2.isnull().sum() # 결측치 끝!

df_flights2



# dep_time / sched_dep_time / arr_time / sched_arr_time 전처리

# float의 경우 int로 변경 후 str로 변경 (데이터타입) -- 전처리 편의를 위해
df_flights2["dep_time"] = df_flights2["dep_time"].astype(int)
df_flights2 #astype() 열 전체의 데이터타입 변환
df_flights2["dep_time"] = df_flights2["dep_time"].astype(str)
len(df_flights2['dep_time'][0]) > 4 
a=0 # 길이가 3이나 4가 아닌 경우
for i in range(len(df_flights2)):
    values = len(df_flights2['dep_time'][i])
    if values > 4:
        a+=1
    elif values < 3:
        a+=1 # a = 876
# 실제 행 확인
list_a =[]
for i in range(len(df_flights2)):
    values = len(df_flights2['dep_time'][i])
    if values > 4:
        list_a.append(i)
    elif values < 3:
        list_a.append(i)

df_flights2['dep_time'][list_a].unique() # 길이가 1또는 2인 행이 존재하고 876개면 그냥 삭제

# 위 과정을 함수로 구현
# str로 바꾼 행의 len()이 3또는 4가 아닌 경우를 카운팅
def check_num(x):
    x = x.astype(int)
    x = x.astype(str)
    a=0 # 길이가 3이나 4가 아닌 경우
    for i in range(len(df_flights2)):
        values = len(x[i])
        if values > 4:
            a+=1
        elif values < 3:
            a+=1 # a = 876
    return a
# str로 바꾼 행의 len()이 3또는 4가 아닌 경우를 직접 확인
def check_num_unique(x):
    x = x.astype(int)
    x = x.astype(str)
    list_a =[]
    for i in range(len(df_flights2)):
        values = len(x[i])
        if values > 4:
            list_a.append(i)
        elif values < 3:
            list_a.append(i)

    return x[list_a].unique()
# 3개 열에서 이상치 발견    
check_num(df_flights2['dep_time']) # 876
check_num(df_flights2['sched_dep_time']) # 0
check_num(df_flights2['arr_time']) # 6864
check_num(df_flights2['sched_arr_time']) # 4125
# 전부 1또는 2의 길이를 가짐
check_num_unique(df_flights2['dep_time'])
check_num_unique(df_flights2['arr_time'])
check_num_unique(df_flights2['sched_arr_time'])

6864 / df_flights2.shape[0] # 전체의 2%라서 이상치로 간주하고 삭제

# 위 4개 열 우선 str로 변환
type(df_flights2['dep_time'][0])
type(df_flights2['sched_dep_time'][0]) # int
df_flights2['sched_dep_time'] = df_flights2['sched_dep_time'].astype(str)
type(df_flights2['arr_time'][0])
df_flights2['arr_time'] = df_flights2['arr_time'].astype(int)
df_flights2['arr_time'] = df_flights2['arr_time'].astype(str)
type(df_flights2['sched_arr_time'][0])
df_flights2['sched_arr_time'] = df_flights2['sched_arr_time'].astype(str)
# 변환 완료
df_flights2.head()

# str로 변환 후 길이가 3보다 작은 행은 지워주는 반복문
str_list = ['dep_time','sched_dep_time','arr_time','sched_arr_time']
set_idx = []
# 해당 열의 인덱스를 set_idx 리스트에 모두 저장을하고 set을 통해 중복 제거
# 이후 drop()이라는 함수를 사용해서 해당 열 제거하고 인덱스 초기화
for c in str_list:
    for i in range(len(df_flights2)):
        if len(df_flights2[c][i]) < 3:
            set_idx.append(i)
std_idx = list(set(set_idx))
df_flights2.shape
df_flights2.drop(std_idx,inplace=True)
df_flights2 = df_flights2.reset_index(drop=True)
df_flights2.shape
df_flights2.head()

# len이 3인 경우 왼쪽 첫번째에 0을 추가해줌
df_flights2['dep_time'] = df_flights2['dep_time'].str.zfill(4)
df_flights2['sched_dep_time'] = df_flights2['sched_dep_time'].str.zfill(4)
df_flights2['arr_time'] = df_flights2['arr_time'].str.zfill(4)
df_flights2['sched_arr_time'] = df_flights2['sched_arr_time'].str.zfill(4)
df_flights2.head()

df_flights2['dep_time_hour'] = df_flights2['dep_time'].str[:2] # 이륙 시간:분 중에 시간
df_flights2['dep_time_minute'] = df_flights2['dep_time'].str[2:] # 분
df_flights2['sched_dep_time_hour'] = df_flights2['dep_time'].str[:2]
df_flights2['sched_dep_time_minute'] = df_flights2['dep_time'].str[2:]
df_flights2['arr_time_hour'] = df_flights2['dep_time'].str[:2]
df_flights2['arr_time_mintute'] = df_flights2['dep_time'].str[2:]
df_flights2['sched_arr_time_hour'] = df_flights2['dep_time'].str[:2]
df_flights2['sched_dep_time_minute'] = df_flights2['dep_time'].str[2:]

# 전처리 완료 데이터 프레임
df_flights2

# plane 데이터프레임과 df_flights 데이터프레임 연결하기
# merged_fligjts 생성
# # year라는 열 이름이 겹쳐서 이름 변경도 해줌
merged_flights = pd.merge(df_flights2,df_planes,on='tailnum',how='left')
merged_flights.rename(columns={'year_x':'flights_year','year_y':'made_year'},inplace=True)

merged_flights.iloc[:,8:].isnull().sum() 
# 결측치가 발생함
# flights_delay에는 있지만 planes에는 없는 tailnum이 존재?
a = merged_flights['tailnum'].unique()
len(a) # 4034 종류
b = df_planes['tailnum'].unique() 
len(b) # 3322 종류
np.isin(a,b).sum() #3313
np.sum(~(np.isin(a,b))) # 721
# 721개의 종류가 flights_delay에만 있음

# 총 몇개의 항공편이 속할까?
(~merged_flights['tailnum'].isin(b)).sum() # 47906개

# 지연시간이 15이상인 행들만 골라서 merged_delay 라는 데이터 프레임 생성
merged_delay = merged_flights[merged_flights['dep_delay']>15]
merged_delay
merged_delay = merged_delay.reset_index()
merged_delay
merged_delay.shape # (120559, 35)
merged_delay.head()

# 제조사가 보잉사인 것들만 남기기
df_planes['manufacturer'].unique()
merged_delay['manufacturer'].unique()
BOEING_delay = merged_delay[merged_delay['manufacturer'] == 'BOEING']
BOEING_delay.reset_index(drop=True,inplace=True)
BOEING_delay.shape
BOEING_delay

# df_flights2 전처리 데이터 프레임
# merged_flights 위 데이터 프레임과 df_planes를 tailnum을 기준으로 left join 한 것
# merged_delay 위 데이터프레임 중 출발 지연이 발생한 행만 모은 데이터프레임
# BOEING_delay 위 데이터프레임 중 보잉만 필터링

# 시각화 라이브러리리
import seaborn as sns
import matplotlib.pyplot as plt    

merged_delay_group = merged_delay.groupby('manufacturer', as_index=False)['dep_delay'].agg(
    mean='mean',
    count='count'
)
merged_delay.shape # 12만개
merged_delay_group

# 제조사별 출발 지연 발생 횟수
sns.barplot(data=merged_delay_group, x='manufacturer', y='count', color='skyblue', label='Departure Delay')
plt.title('delay counts')
plt.ylabel('count')
plt.legend()
plt.show()
# 제조사별 출발 지연 발생 빈도
# 전체 제조사별 항공편 개수 
total_group = merged_flights.groupby('manufacturer', as_index=False)['dep_delay'].count()
total_group = total_group.rename(columns={'dep_delay': 'total_count'})
# 두 데이터프레임 병합
merged_stats = pd.merge(merged_delay_group, total_group, on='manufacturer', how='left')
# 비율 열 새로 생성
merged_stats['delay_ratio'] = merged_stats['count'] / merged_stats['total_count']
merged_stats.sort_values('delay_ratio',ascending=False)
# 보잉은 79922개중 19프로가 지연인데
# emb는 61000개중 25프로가 지연임
# 시각화
sns.barplot(data=merged_stats, x='manufacturer', y='delay_ratio', color='skyblue', label='Departure Delay')
plt.title('delay ratio')
plt.ylabel('ratio')
plt.legend()
plt.show()
# 전체 중 보잉만
merged_boeing = merged_flights[merged_flights['manufacturer'] == 'BOEING']
merged_boeing = merged_boeing.reset_index()
merged_boeing['dest'].value_counts()
merged_boeing['origin'].value_counts()

# 평균낼 때 너무 높은 값이 포함
sns.histplot(merged_delay['dep_delay'], bins=100, kde=True)
plt.title('dep_delay')
plt.show()
merged_delay['dep_delay'].max()

# 출발 지연에 영향을 주는 요소가 분명히 있음
# 도착 공항에 따라일 수도 있고
# 거리일 수도 있고
# 시간대 일 수도 있고

# 결론적으로 보잉 항공사와 EMBRAER항공사가 어떤 차이점이 있길래 지연율에서 차이가 날까?

# 궁금증1. 1년 중 특정 달에 많은 항공편을 보유하고 있는 부분에서 차이가 날까?
boeing_group = merged_boeing.groupby('month',as_index=False)[['dep_delay']].count()
boeing_group.rename(columns={'dep_delay':'counts'},inplace=True)
sns.barplot(data=boeing_group, x='month', y='counts', color='skyblue', label='Departure Delay')
plt.title('boeing monthly counts')
plt.ylabel('counts')
plt.legend()
plt.show()
# 전체 중 EMBRAER만
merged_emb = merged_flights[merged_flights['manufacturer'] == 'EMBRAER']
emb_group = merged_emb.groupby('month',as_index=False)[['dep_delay']].count()
emb_group.rename(columns={'dep_delay':'counts'},inplace=True)
sns.barplot(data=emb_group, x='month', y='counts', color='skyblue', label='Departure Delay')
plt.title('emb monthly counts')
plt.ylabel('counts')
plt.legend()
plt.show() # 엄청난 차이를 보이진 않지만 9,10,11월에 조금 적음 emb가


merged_flights['dest'].value_counts()
# 보잉사 출발 도착 공항 
# 보잉사는 EWR 공항에서 주로 출발함
merged_boeing['dest'].value_counts()
merged_boeing['origin'].value_counts()
#EWR 공항 출발
merged_boeing[merged_boeing['origin']=="EWR"]['dest'].value_counts()
#JFK
merged_boeing[merged_boeing['origin']=="JFK"]['dest'].value_counts()
#LGA
merged_boeing[merged_boeing['origin']=="LGA"]['dest'].value_counts()

# EWR 공항에서 보잉사가 주로 출발하는데 EWR의 평균 지연시간이 가장 높음
merged_flights.groupby('origin')['dep_delay'].agg(mean = 'mean', count= 'count')

# 공항별로 계획된 항공편의 특징이 있을까? 거리라든가 SEATS 비행기 크기라던가

# 거리의 경우 jfk가 가장 1274로 가장 멀리 이동하는 항공기들이 있고 lga가 783으로 짧은 항공편 위주
merged_flights.groupby('origin')['distance'].agg(mean = 'mean', count= 'count')
# 공항 별 수용하는 비행기의 좌석수에는 차이가 있을까
# ewr 항공편 좌석개수 평균 # 150개
EWR_flights = merged_flights[merged_flights['origin']=='EWR']
ewr_model = EWR_flights['model'].unique()
df_planes[df_planes['model'].isin(ewr_model)]['seats'].mean()
# jfk 항공편 좌석개수 평균 # 168개 
JFK_flights = merged_flights[merged_flights['origin']=='JFK']
JFK_model = JFK_flights['model'].unique()
df_planes[df_planes['model'].isin(JFK_model)]['seats'].mean()
# lga 항공편 좌석개수 평균 # 143개
LGA_flights = merged_flights[merged_flights['origin']=='LGA']
LGA_model = LGA_flights['model'].unique()
df_planes[df_planes['model'].isin(LGA_model)]['seats'].mean()
# 좌석수의 경우도 JFK가 가장 크고 LGA가 가장 적음

# JFK는 장거리 비행 + 큰 여객기 // LGA는 단거리 비행 + 작은 여객기

# 전체 중 EMBRAER 만
merged_embraer = merged_flights[merged_flights['manufacturer'] == 'EMBRAER']
merged_embraer = merged_embraer.reset_index()
merged_embraer.groupby('origin')['dep_delay'].agg(mean = 'mean', count= 'count') # ewr 17 jfk 11 lga 1
# 보잉사랑 비교
merged_boeing.groupby('origin')['dep_delay'].agg(mean = 'mean', count= 'count') # ewr 11 jfk 6 lga 13
merged_flights['origin'].value_counts()




# 궁금증2. 시간대에 차이가 있을까?
ewa_group =

#

merged_flights[merged_flights['origin']== "JFK"]['manufacturer'].value_counts()
merged_flights[merged_flights['origin']== "LGA"]['manufacturer'].value_counts()
merged_flights[merged_flights['origin']== "EWR"]['manufacturer'].value_counts()




