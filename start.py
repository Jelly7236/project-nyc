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
















    















# 오래된 것일수록 비행시간이 늘어나는가?
# 딜레이타임 원인 분석  

# --- 딜레이가 덜 되는 항공사 or 공항
# --- 정시운항이 잘 되는지
# --- 딜레이가 가자 잘 일어나는 패턴-주기
# --- 항공편이 많을수록 딜레이가 자주 일어나는지 >> 공항별 항공편 수
# --- 항공사 별 비행기 수 / 좌석 수
# --- 위도 경도 별 날씨 특징 파악해서 딜레이
# --- 가징 크리티컬하게 영향을 주는 날씨의 특징 
# --- 역풍인지 아닌지를 체크해서 비행시간의 차이를 체크해보자
# --- 성수기일때 실제로 딜레이가 많이 이루어지는지 -- 덜 딜레이되는 최적의 항공사 찾기
# --- 공항마다 딜레이가 가장 잘 되는 시간대

# 늦게 도착하는 3가지 경우를 분석하자
# 1. 일찍 출발해서 2. 정시 출발 3. 늦게 출발
# 데이터프레임 3개로 나누어서 우선 분석
# 정시 출발의 범위를 우선적으로 설정 
even_start = df_flights[df_flights['dep_delay']==0] # 정시출발이 1만6천
early_start = df_flights[df_flights['dep_delay']<0] # 일찍 출발 18만3천 10분 일찍 출발까지는 정상 출발로 인정
late_start = df_flights[df_flights['dep_delay']>0] # 연착 12만


# 해진이형 의견 -- 연착 또는 결항 시 항공사가 부담해야 하는 보상 비용을 계산하자
# 단기, 중기, 장기, 결항으로 나누어서 고려