import json
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns


def get_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_group_data(group_key: str, full_data: dict) -> dict:
    result = {}
    for key, value in full_data.items():
        if isinstance(value, dict) and group_key in value:
            result[key] = value[group_key]
    return result


class MonthlyAverageAnalyzer:
    def __init__(self):
        self.data = None
        self.monthly_averages = None

    def load_data(self, periods, values):
        """
        데이터를 로드하고 전처리

        Args:
            periods: 기간 리스트 (예: ['22.04', '22.05', ...])
            values: 값 리스트
        """
        # 데이터프레임 생성
        self.data = pd.DataFrame({"period": periods, "value": values})

        # 년도와 월 분리
        self.data["year"] = self.data["period"].str[:2].astype(int) + 2000
        self.data["month"] = self.data["period"].str[3:].astype(int)

        # 날짜 컬럼 생성
        self.data["date"] = pd.to_datetime(
            self.data["year"].astype(str)
            + "-"
            + self.data["month"].astype(str).str.zfill(2)
            + "-01"
        )

        return self.data

    def calculate_monthly_averages(self):
        """월별 평균값 계산"""
        self.monthly_averages = (
            self.data.groupby("month")
            .agg({"value": ["mean", "count", "std", "min", "max"]})
            .round(2)
        )

        # 컬럼명 정리
        self.monthly_averages.columns = ["average", "count", "std", "min", "max"]
        self.monthly_averages = self.monthly_averages.reset_index()

        # 월 이름 추가
        month_names = [
            "1월",
            "2월",
            "3월",
            "4월",
            "5월",
            "6월",
            "7월",
            "8월",
            "9월",
            "10월",
            "11월",
            "12월",
        ]
        self.monthly_averages["month_name"] = self.monthly_averages["month"].apply(
            lambda x: month_names[x - 1]
        )

        return self.monthly_averages

    def plot_monthly_averages(self, figsize=(12, 6), save_path=None):
        """월별 평균값 시각화"""
        if self.monthly_averages is None:
            self.calculate_monthly_averages()

        # 한글 폰트 설정 (시스템에 따라 조정 필요)
        plt.rcParams["font.family"] = ["AppleGothic"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=figsize)

        # 선 그래프
        ax.plot(
            self.monthly_averages["month"],
            self.monthly_averages["average"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2563eb",
        )

        # 그래프 스타일링
        ax.set_xlabel("월", fontsize=12)
        ax.set_ylabel("평균값", fontsize=12)
        ax.set_title("월별 평균값 트렌드", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # x축 레이블
        ax.set_xticks(self.monthly_averages["month"])
        ax.set_xticklabels([f"{m}월" for m in self.monthly_averages["month"]])

        # 값 표시
        for i, row in self.monthly_averages.iterrows():
            ax.annotate(
                f'{row["average"]:.1f}',
                (row["month"], row["average"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_comparison(self, figsize=(15, 8), save_path=None):
        """원본 데이터와 월별 평균 비교 시각화"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # 원본 데이터 플롯
        ax1.plot(
            range(len(self.data)),
            self.data["value"],
            marker="o",
            linewidth=1,
            markersize=4,
            alpha=0.7,
            color="gray",
        )
        ax1.set_title("원본 시계열 데이터", fontsize=12, fontweight="bold")
        ax1.set_ylabel("값")
        ax1.grid(True, alpha=0.3)

        # x축 레이블 (일부만 표시)
        tick_indices = range(0, len(self.data), max(1, len(self.data) // 10))
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(
            [self.data.iloc[i]["period"] for i in tick_indices], rotation=45
        )

        # 월별 평균 플롯
        ax2.plot(
            self.monthly_averages["month"],
            self.monthly_averages["average"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2563eb",
        )
        ax2.set_title("월별 평균값", fontsize=12, fontweight="bold")
        ax2.set_xlabel("월")
        ax2.set_ylabel("평균값")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(self.monthly_averages["month"])
        ax2.set_xticklabels([f"{m}월" for m in self.monthly_averages["month"]])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def get_summary_stats(self):
        """요약 통계 반환"""
        if self.monthly_averages is None:
            self.calculate_monthly_averages()

        summary = {
            "최고_평균_월": self.monthly_averages.loc[
                self.monthly_averages["average"].idxmax(), "month_name"
            ],
            "최고_평균값": self.monthly_averages["average"].max(),
            "최저_평균_월": self.monthly_averages.loc[
                self.monthly_averages["average"].idxmin(), "month_name"
            ],
            "최저_평균값": self.monthly_averages["average"].min(),
            "전체_평균": self.monthly_averages["average"].mean().round(2),
            "표준편차": self.monthly_averages["average"].std().round(2),
        }

        return summary

    def print_detailed_report(self):
        """상세 리포트 출력"""
        if self.monthly_averages is None:
            self.calculate_monthly_averages()

        print("=" * 50)
        print("월별 평균값 분석 리포트")
        print("=" * 50)

        # 월별 상세 정보
        for _, row in self.monthly_averages.iterrows():
            print(
                f"{row['month_name']:>3}: 평균 {row['average']:>6.2f} "
                f"(개수: {row['count']:>2}, 표준편차: {row['std']:>6.2f})"
            )

        print("-" * 50)

        # 요약 통계
        summary = self.get_summary_stats()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ')}: {value}")


#     # 분석 실행
#     analyzer = MonthlyAverageAnalyzer()

#     # 데이터 로드
#     data = analyzer.load_data(periods, values)
#     print("데이터 로드 완료")
#     print(f"총 {len(data)}개 데이터 포인트")

#     # 월별 평균 계산
#     monthly_avg = analyzer.calculate_monthly_averages()
#     print("\n월별 평균값 계산 완료")

#     # 상세 리포트 출력
#     analyzer.print_detailed_report()

#     # 시각화
#     analyzer.plot_monthly_averages()
#     analyzer.plot_comparison()

#     # CSV 저장 (선택사항)
#     # monthly_avg.to_csv('monthly_averages.csv', index=False, encoding='utf-8-sig')
#     # print("\n월별 평균값이 'monthly_averages.csv'에 저장되었습니다.")


def get_data_from_txt(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line.strip())
            data_list.append(line_data)
    return data_list


def get_biz_lst(file_path):
    categories = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",") if x.strip()]
            categories.extend(parts)  # 리스트에 이어붙이기
    return categories


# 2. 엑셀 파일에 컬럼 추가
def get_exel_with_biz_lst(txt_path, xlsx_path, output_path):
    # 분류 결과 불러오기
    categories = get_biz_lst(txt_path)

    # 엑셀 데이터 불러오기
    df = pd.read_excel(xlsx_path)

    # 길이 확인
    if len(df) != len(categories):
        raise ValueError(
            f"❌ 엑셀 행 수 ({len(df)})와 분류 리스트 길이 ({len(categories)})가 다릅니다!"
        )

    # 새 컬럼 추가
    df["그룹"] = categories

    # 저장
    df.to_excel(output_path, index=False)
    print(f"✅ 새로운 파일이 저장되었습니다: {output_path}")
