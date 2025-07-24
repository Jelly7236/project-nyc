#ev 항공사 기체 촤석 분포
# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# EV 항공사 좌석 정보 준비
EV_seats = EV_total[['seats']].dropna()

# 좌석 구간 설정
bins = [0, 50, 70, 90, 110, 130, 150, float('inf')]
labels = ['~50석', '51~70석', '71~90석', '91~110석', '111~130석', '131~150석', '151석 이상']
EV_seats['좌석구간'] = pd.cut(EV_seats['seats'], bins=bins, labels=labels, right=False)

# 좌석 구간별 비율 계산
seat_distribution = (
    EV_seats['좌석구간']
    .value_counts(normalize=True)
    .sort_index() * 100
)

# 시각화
plt.figure(figsize=(10, 6))
colors = sns.color_palette('pastel', len(seat_distribution))
barplot = sns.barplot(x=seat_distribution.index, y=seat_distribution.values, palette=colors)

# y축 범위 명시적으로 지정 (최대 100)
plt.ylim(0, 100)

# 제목 및 축 설정
plt.title('EV 항공사 기체 좌석 수 분포', fontsize=16, fontweight='bold')
plt.xlabel('좌석 수 구간', fontsize=12)
plt.ylabel('비율 (%)', fontsize=12)

# 수치 라벨 (막대 위에 충분히 띄워서)
for i, value in enumerate(seat_distribution.values):
    plt.text(i, value + 2, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()