import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 경로
file_path = "./data2_usage.xlsx"  # 로컬 경로로 바꿔주세요

# 데이터 불러오기
df = pd.read_excel(file_path)

# 월 컬럼 리스트
month_cols = [f"사용량_{i}월" for i in range(1, 13)]
month_labels = [f"{i}월" for i in range(1, 13)]

# 그룹(=용도)별 평균 계산
df_grouped = df.groupby("용도")[month_cols].mean()

# 상위 5개 그룹만 시각화 (원하면 전체로 변경 가능)
top_groups = df_grouped.index
plt.rcParams["font.family"] = ["AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
# 그래프 그리기
plt.figure(figsize=(12, 6))
for group in top_groups:
    plt.plot(month_labels, df_grouped.loc[group], label=group)

plt.title("용도별 월별 평균 사용량")
plt.xlabel("월")
plt.ylabel("평균 사용량")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
