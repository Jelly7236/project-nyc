# ✈️ NYC 항공편 지연 분석: ExpressJet Airlines(EV) 집중 분석

## 📊 프로젝트 개요

뉴욕 3대 공항(JFK, LGA, EWR)에서 출발하는 항공편 데이터를 분석하여 **ExpressJet Airlines(EV)의 높은 지연율 원인**을 규명하고 해결방안을 제시하는 데이터 분석 프로젝트입니다.

### 🎯 핵심 발견사항
- **EV 항공사의 지연율이 41.2%**로 UA(29.8%), B6(39.0%)보다 현저히 높음
- EV는 **소형 항공기 중심의 단거리 노선** 운영 > 날씨의 영향을 더욱 예민하게 받음
- **낮은 기체 회전율**과 **특정 공항에서 관제 우선 순위 밀리는 것**이 주요 원인

## 🛠 사용 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)

```python
# 주요 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nycflights13 as flights
```

## 📁 데이터셋

**nycflights13** 패키지의 2013년 뉴욕 항공편 데이터 활용
- `flights`: 항공편 정보 (336,776건)
- `airlines`: 항공사 정보
- `airports`: 공항 정보  
- `planes`: 항공기 정보
- `weather`: 날씨 정보

## 🔍 분석 프로세스

### 0. 지연 기준 정의
- 출발 지연과 도착 지연 중 출발 지연만 고려 ( 명확한 원인을 분석하기 위함 )
- 15분 이상 지연된 경우만 실제 지연으로 분류류

### 1. 데이터 전처리
```python
# 결측치 제거 및 데이터 정제
flights_cleaned = df_flights.dropna(subset=['dep_time','dep_delay', 'arr_time','arr_delay','tailnum', 'air_time'])
# 15분 이상 지연 기준 설정
flights_delay = flights_cleaned[flights_cleaned['dep_delay']>=15]
```

### 2. 탐색적 데이터 분석 (EDA)

#### 📈 항공사별 운항량 및 지연율 분석
- **운항량 TOP 3**: UA(57,782편) → B6(54,049편) → EV(51,108편)
- **지연율 비교**: EV(31.03%) > B6(23.43%) > UA(21.75%)

#### 🏭 항공기 제조사 분석
| 항공사 | 주요 제조사 | 평균 좌석수 |
|--------|-------------|-------------|
| UA | Boeing (74.6%) | 176석 |
| B6 | Airbus (57.9%) | 134석 |
| **EV** | **Embraer (69.3%)** | **59석** |

#### 🛫 노선 거리 분석
- **EV**: 단거리(92.2%) 중심의 리저널 항공사
- **UA/B6**: 중장거리 노선까지 다양하게 운영
- <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/b472104e-92e1-4f2d-950f-3a5992932d16" />


### 3. 지연 원인 심층 분석

#### 🔄 기체 회전율 분석
- 회전율 구한 방법
  - 동일한 모델명으로 GROUPBY를 진행함
  - DIFF() 함수를 사용하여 위아래 행들의 값에 대한 차를 구해줌
  - time_gap이 시간 차이(Timedelta)이기 때문에 dt.total_seconds()를 사용하여 초 단위로 변환 후 3600으로 나누어 **시간 단위(float)**로 바꾸어줌
    결과: gap_hours에는 **기체별 비행 간 간격(시간)**이 들어감.

```python
# 기체별 평균 비행 간격 계산
ev_schedule['time_gap'] = ev_schedule.groupby('tailnum')['month_day_time'].diff()
ev_schedule['gap_hours'] = ev_schedule['time_gap'].dt.total_seconds() / 3600
```
<img width="990" height="490" alt="image" src="https://github.com/user-attachments/assets/ce5559f4-7e67-4a36-98b5-3a69494dfbfa" />

**핵심 발견**: EV의 소형 항공기들이 상대적으로 **낮은 회전율**을 보임 > 특정 기체들만 반복적으로 비행을 반복함

#### 🏢 공항별 특성 분석
- **JFK 공항**: 평균 거리 1,222마일 (장거리 중심)
- **LGA 공항**: 평균 거리 872마일 (단거리 중심)
- EV는 JFK에서 장거리 노선 운영 시 더 큰 지연 발생

## 📊 주요 시각화

### 1. 항공사별 지연율 비교
```python
plt.figure(figsize=(8,6))
sns.barplot(data=summary_big3, x='carrier', y='delay_rate (%)', palette=bar_colors)
plt.title('항공사별 출발 지연 비율(%)', fontsize=18, fontweight='bold')
```

### 2. 월별 지연 패턴 분석
- 여름철(6-8월) 지연율 증가 패턴 확인
- EV의 계절별 지연율 변동성이 가장 큼

### 3. 제조사별 항공기 특성
```python
# 파이차트로 제조사 비율 시각화
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
```

## 💡 결론 및 개선방안

### 🔍 지연 원인
1. **소형 항공기의 운영 특성**: 55.6석 평균으로 탑승객 처리 시간 증가
2. **낮은 기체 회전율**: 효율적인 기체 운용 부족
3. **공항별 노선 특성**: JFK의 장거리 노선에서 높은 지연율

### 🚀 개선 제안
1. **기체 회전율 최적화**: 스케줄링 알고리즘 개선
2. **공항별 차별화 전략**: JFK 장거리 vs LGA 단거리 특화
3. **예측 모델 구축**: 지연 위험도 예측 시스템 도입

## 📈 프로젝트 성과

- **데이터 전처리**: 336K → 327K 항공편 데이터 정제
- **시각화**: 15+ 차트를 통한 인사이트 도출  
- **통계 분석**: 3개 항공사 다각도 비교 분석
- **비즈니스 인사이트**: 실무진이 활용 가능한 구체적 개선방안 제시

## 🔗 파일 구조

```
flight-delay-analysis/
├── data/
│   └── nycflights13 데이터
├── notebooks/
│   └── flight_analysis.ipynb
├── src/
│   └── analysis_script.py
├── visualizations/
│   ├── delay_rate_comparison.png
│   ├── monthly_pattern.png
│   └── manufacturer_analysis.png
└── README.md
```

## 🎓 학습 포인트

- **대용량 데이터 핸들링**: 30만+ 레코드 효율적 처리
- **비즈니스 인사이트 도출**: 데이터에서 실무적 해결방안 제시
- **시각화 스토리텔링**: 복잡한 분석 결과의 직관적 전달
- **통계적 사고**: 가설 설정 → 검증 → 결론 도출 프로세스

---

### 📞 연락처
- **Email**: [your.email@domain.com]
- **LinkedIn**: [LinkedIn Profile]
- **Portfolio**: [Portfolio Website]

> 이 프로젝트는 데이터 분석을 통한 비즈니스 문제 해결 능력을 보여주는 포트폴리오입니다.
