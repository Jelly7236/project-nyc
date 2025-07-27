import pandas as pd
import nycflights13 as flights
import numpy as np

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

df_flights.shape # 3 해진 희재 소영
df_airlines.shape # 16
df_airports.shape # 1458 -a 우영
df_planes.shape # 3322 -a 우영
df_weather.shape # 26115 -b 유진

# df_fights 구성 확인
df_flights.info()
df_flights.shape # (336776, 19)

# 'hour', 'minute'은 'sched_dep_time'을 나누어서 추출됨

# 결측치 처리
df_flights.isnull().sum() # 결측치가 몇개 있는지를 확인..

# dep_time(실제 이륙 시간)과 dep_delay(이륙 지연 시간)의 null값 갯수가 같음 825

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
df_flights2.columns


##############################################
'''
delay 측면

delay 조건
1. 일찍 출발 => 늦게 도착
2. 정시 출발 => 늦게 도착
3. 늦게 출발 => 늦게 도착 

delay 기준

기본적으로 1시간 이상을 기준으로 한다=> 1시간 이내 그냥 보상 X 

<국제선 기준>
1. 하 => 2시간 이상 4시간 미만 - 운임의 10%
2. 중 => 4시간 이상 12시간 미만 - 운임의 20%
3. 상 => 12시간 이상 - 운임의 30% 
라고 가정하면, 

4. 보상 제외 => 천재지변, 기상 악화, 항공기 정비 등 불가항력적인 사유

delay가 많을수록 항공사가 보상을 해야하기 때문에, 항공사 손실 증가 => * 가장 손실이 많이 발생하는 항공사는? 

'''

# 1. 일찍 출발 => 늦게 도착
early_dep=((df_flights2['arr_delay']>0) & (df_flights2['dep_delay']<0))
sum(early_dep) #35084

# 2. 정시 출발 => 늦게 도착
even_dep=((df_flights2['arr_delay']>0) & (df_flights2['dep_delay']==0))
sum(even_dep) #4926 

# 3. 늦게 출발 => 늦게 도착
late_dep=(df_flights2['arr_delay']>0) & (df_flights2['dep_delay']>0)
sum(late_dep) #85769

df_flights2['early_dep'] = early_dep
df_flights2['even_dep'] = even_dep
df_flights2['late_dep'] = late_dep


# 항공사(carrier)별 delay case 
carrier_grp=df_flights2.groupby('carrier')[['early_dep','even_dep','late_dep']].sum()

# 항공사(carrier)별 total dealy
carrier_grp['total_delay'] = (
    carrier_grp['early_dep'] +
    carrier_grp['even_dep'] +
    carrier_grp['late_dep']
)


'''
항공사 규모 분류하기

1. 항공편 개수 => 많을 수록 규모가 큰 항공사라고 판단 가능한가?

2. 비행 거리 => 비행 거리가 길수록 규모가 큰 항공사라고 판단 가능한가?

'''
# 항공사 별 항공편 개수로 판단하기 (내림차순)
count = df_flights2.groupby('carrier').size()
count.sort_values(ascending=False)

# 항공사 별 비행 거리로 판단하기 (내림차순)
dist = (df_flights2.groupby('carrier')['distance']).sum()
dist.sort_values(ascending=False)

carrier_grp['count_flight']=count
carrier_grp['dist_flight']=dist


carrier_grp = carrier_grp[['early_dep','even_dep','late_dep','total_delay','count_flight','dist_flight']]

# delay와 항공편 수, 비행 거리를 나타낸 데이터프레임
carrier_grp
carrier_grp.sort_values(by=['dist_flight', 'count_flight'], ascending=[False, False])


'''
delay측면과 항공사 규모 측면을 생각했을때,

=> dist, count 의 값이 클수록 규모가 큰 항공사라고 판단 가능한가?
    => 노선 별 거리(단거리/ 중거리/ 장거리)인지를 판단 후 count를 체크해야할 것 같음
    
    => 거리를 나누는 기준
    유럽 항공 안전 기구(Eurocontrol)는 500km 미만을 "매우 단거리", 500~1,500km를 "단거리", 1,500~4,000km를 "중거리", 4,000km 이상을 "장거리"로 정의합니다.
    => df_flights2['distance'].describe() 해보니까 최대가 4983
    평균이 1044여서 의미 X 생각 

    => 비행 시간(air time)
    6시간 초과는 장거리 
    => 마찬가지아닌가

'''
# total delay / count 의 값이 클수록 손해 규모가 크다고 판단 가능한가?

loss=(carrier_grp['total_delay'] / carrier_grp['count_flight'])*100
loss.sort_values(ascending=False)

carrier_grp['loss']=loss.round(3)
carrier_grp

# loss까지 추가해본 데이터프레임
carrier_grp.sort_values(by=['loss','dist_flight', 'count_flight'], ascending=[False,False, False])


