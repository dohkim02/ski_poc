import json
import pandas as pd
import ast

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
        """월별 중위수값 계산 (median과 IQR 기준)"""
        # 월별 그룹화하여 통계 계산
        monthly_stats = (
            self.data.groupby("month")["value"]
            .agg(
                [
                    "median",  # 중위수
                    "count",  # 개수
                    lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
                    "min",  # 최소값
                    "max",  # 최대값
                ]
            )
            .round(2)
        )

        # 컬럼명 정리 (기존 포맷 유지)
        monthly_stats.columns = ["average", "count", "std", "min", "max"]
        self.monthly_averages = monthly_stats.reset_index()

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
        """월별 중위수값 시각화"""
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
        ax.set_ylabel("중위수값", fontsize=12)
        ax.set_title("월별 중위수값 트렌드", fontsize=14, fontweight="bold")
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
        """원본 데이터와 월별 중위수 비교 시각화"""
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

        # 월별 중위수 플롯
        ax2.plot(
            self.monthly_averages["month"],
            self.monthly_averages["average"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2563eb",
        )
        ax2.set_title("월별 중위수값", fontsize=12, fontweight="bold")
        ax2.set_xlabel("월")
        ax2.set_ylabel("중위수값")
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
            "최고_중위수_월": self.monthly_averages.loc[
                self.monthly_averages["average"].idxmax(), "month_name"
            ],
            "최고_중위수값": self.monthly_averages["average"].max(),
            "최저_중위수_월": self.monthly_averages.loc[
                self.monthly_averages["average"].idxmin(), "month_name"
            ],
            "최저_중위수값": self.monthly_averages["average"].min(),
            "전체_중위수": self.monthly_averages["average"].median().round(2),
            "IQR": (
                self.monthly_averages["average"].quantile(0.75)
                - self.monthly_averages["average"].quantile(0.25)
            ).round(2),
        }

        return summary

    def print_detailed_report(self):
        """상세 리포트 출력"""
        if self.monthly_averages is None:
            self.calculate_monthly_averages()

        print("=" * 50)
        print("월별 중위수값 분석 리포트")
        print("=" * 50)

        # 월별 상세 정보
        for _, row in self.monthly_averages.iterrows():
            print(
                f"{row['month_name']:>3}: 중위수 {row['average']:>6.2f} "
                f"(개수: {row['count']:>2}, IQR: {row['std']:>6.2f})"
            )

        print("-" * 50)

        # 요약 통계
        summary = self.get_summary_stats()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ')}: {value}")


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
# 그룹 컬럼이 포함된 엑셀 파일 생성
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


import json


def find_group_usage_combination(data, target_grade, target_group, target_usage):
    """
    JSON 파일에서 특정 그룹, 용도, 등급 조합을 찾는 함수

    Args:
        target_group: 찾고자 하는 그룹명 (예: '제조업')
        target_usage: 찾고자 하는 용도명 (예: '일반용1')
        target_grade: 찾고자 하는 등급 (예: 'A', 선택적)

    Returns:
        list: 매칭되는 인덱스들의 리스트
    """
    groups = data["그룹"]
    usages = data["용도"]

    # 1. 그룹에서 target_group과 일치하는 인덱스들 찾기
    group_indices = []
    for index, group in groups.items():
        if group == target_group:
            group_indices.append(index)

    # 2. 용도에서 target_usage와 일치하는 인덱스들 찾기
    usage_indices = []
    for index, usage in usages.items():
        if usage == target_usage:
            usage_indices.append(index)

    # 3. 등급이 지정된 경우 등급도 확인
    grade_indices = []
    if target_grade and "등급" in data:
        grades = data["등급"]
        for index, grade in grades.items():
            if grade == target_grade:
                grade_indices.append(index)

    # 4. 교집합 구하기
    if target_grade and grade_indices:
        # 등급이 지정되고 데이터에 등급이 있는 경우 3개 조건 모두 만족
        matching_indices = list(
            set(group_indices) & set(usage_indices) & set(grade_indices)
        )
    else:
        # 등급이 지정되지 않았거나 데이터에 등급이 없는 경우 기존 방식
        matching_indices = list(set(group_indices) & set(usage_indices))

    return matching_indices


def get_group_usage_info(data, target_grade, target_group, target_usage):
    """
    특정 그룹, 용도, 등급 조합의 모든 정보를 반환하는 함수

    Args:
        data: JSON 데이터
        target_group: 찾고자 하는 그룹명
        target_usage: 찾고자 하는 용도명
        target_grade: 찾고자 하는 등급
    """
    matching_indices = find_group_usage_combination(
        data, target_grade, target_group, target_usage
    )

    if not matching_indices:
        grade_info = f", 등급: '{target_grade}'" if target_grade else ""
        return (
            f"'{target_group}', '{target_usage}'{grade_info} 조합을 찾을 수 없습니다."
        )

    results = []
    for index in matching_indices:
        result_item = {
            "index": index,
            "grade": data["등급"][index],
            "group": data["그룹"][index],
            "usage": data["용도"][index],
            "data_num": data["데이터 개수"][index],
        }
        results.append(result_item)
    # 카테고리 정보 생성

    category_parts = [results[0]["grade"], results[0]["group"], results[0]["usage"]]

    category = "(" + ", ".join(category_parts) + ")"
    data_num = results[0]["data_num"]

    return {
        "category": category,
        "standard": data["사용량 패턴 기준값"][results[0]["index"]],
        "data_num": data_num,
    }

    # return {
    #     "category": category,
    #     "median": data["사용량 패턴 중앙값"][results[0]["index"]],
    #     "data_num": data_num,
    # }
