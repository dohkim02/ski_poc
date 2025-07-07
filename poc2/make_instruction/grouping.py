import pandas as pd
import json
import ast
from collections import defaultdict
import numpy as np
import os
import sys
from scipy import stats
from scipy.stats import jarque_bera, skew, kurtosis

data_path = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(data_path)

# utils 모듈에서 열량 범위 분류 함수 import
from utils import get_heat_input_gt


class ExcelGroupProcessor:
    def __init__(self, file_path):
        """
        엑셀 파일을 읽어서 초기화

        Args:
            file_path (str): 엑셀 파일 경로
        """
        self.file_path = file_path
        self.df = None
        self.grouped_data = None

    def load_data(self):
        """엑셀 파일을 읽어오기"""
        try:
            # 엑셀 파일 확장자에 따라 다르게 처리
            if self.file_path.endswith(".csv"):
                self.df = pd.read_csv(self.file_path, encoding="utf-8")
            else:
                self.df = pd.read_excel(self.file_path)

            print(f"데이터 로드 완료: {len(self.df)}행, {len(self.df.columns)}열")
            print("컬럼명:", list(self.df.columns))
            return True

        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return False

    def parse_usage_pattern(self, pattern_str):
        """
        사용량_패턴 문자열을 파싱하여 딕셔너리로 변환

        Args:
            pattern_str (str): 사용량 패턴 문자열

        Returns:
            dict: 월별 사용량 딕셔너리
        """
        try:
            # 문자열이 딕셔너리 형태인 경우
            if isinstance(pattern_str, str):
                # 작은따옴표를 큰따옴표로 변경하여 JSON 파싱 가능하게 만들기
                pattern_str = pattern_str.replace("'", '"')
                return json.loads(pattern_str)
            elif isinstance(pattern_str, dict):
                return pattern_str
            else:
                return {}
        except:
            try:
                # ast.literal_eval을 사용한 파싱 시도
                return ast.literal_eval(str(pattern_str))
            except:
                print(f"사용량 패턴 파싱 실패: {pattern_str}")
                return {}

    def remove_outliers_iqr(self, values, multiplier=1.5):
        """
        IQR 방법을 사용하여 이상치 제거

        Args:
            values (list): 값들의 리스트
            multiplier (float): IQR 배수 (기본값: 1.5)

        Returns:
            list: 이상치가 제거된 값들의 리스트
        """
        if len(values) < 4:  # 값이 너무 적으면 이상치 제거 안 함
            return values

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        filtered_values = [v for v in values if lower_bound <= v <= upper_bound]

        # 이상치가 발견된 경우 정보 출력
        if len(filtered_values) < len(values):
            removed_count = len(values) - len(filtered_values)
            print(
                f"         - 이상치 {removed_count}개 제거됨 (전체 {len(values)}개 중)"
            )

        return filtered_values

    def analyze_distribution(self, values):
        """
        데이터의 분포를 분석하여 최적의 통계량 결정

        Args:
            values (list): 분석할 값들의 리스트

        Returns:
            dict: 분포 분석 결과와 권장 통계량
        """
        if len(values) < 3:
            return {
                "method": "median",
                "reason": "insufficient_data",
                "normality_p": None,
                "skewness": None,
                "mad_ratio": None,
            }

        values = np.array(values)

        # 정규성 검정 (Jarque-Bera test)
        try:
            jb_stat, jb_p = jarque_bera(values)
        except:
            jb_p = 0.0

        # 치우침 측정
        skewness = skew(values)

        # MAD 계산 및 변동성 측정
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        mad_ratio = mad / median_val if median_val != 0 else 0

        # 결정 로직
        if jb_p > 0.05:  # 정규분포에 가까움
            method = "trimmed_mean"
            reason = "normal_distribution"
        elif abs(skewness) > 1.0:  # 치우친 분포
            method = "median"
            reason = "skewed_distribution"
        elif mad_ratio > 0.5:  # 변동성이 큼
            method = "mad_adjusted"
            reason = "high_variability"
        else:  # 기본값
            method = "trimmed_mean"
            reason = "default_choice"

        return {
            "method": method,
            "reason": reason,
            "normality_p": jb_p,
            "skewness": skewness,
            "mad_ratio": mad_ratio,
        }

    def calculate_trimmed_mean(self, values, trim_percent=0.1):
        """
        절삭평균 계산

        Args:
            values (list): 값들의 리스트
            trim_percent (float): 절삭 비율 (각 끝에서 제거할 비율)

        Returns:
            float: 절삭평균
        """
        return stats.trim_mean(values, trim_percent * 2)  # scipy는 양쪽 합계를 받음

    def calculate_mad_adjusted_center(self, values):
        """
        MAD 기반 조정된 중심값 계산

        Args:
            values (list): 값들의 리스트

        Returns:
            float: MAD 조정된 중심값
        """
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))

        # MAD 기반 robust한 중심값 계산
        # 중앙값 주변 1.5*MAD 범위의 값들만 사용하여 평균 계산
        mad_threshold = 1.5 * mad
        robust_values = [v for v in values if abs(v - median_val) <= mad_threshold]

        if len(robust_values) > 0:
            return np.mean(robust_values)
        else:
            return median_val

    def calculate_adaptive_monthly_stats(self, usage_patterns, remove_outliers=False):
        """
        분포 분석 기반 적응형 월별 통계 계산

        Args:
            usage_patterns (list): 사용량 패턴 딕셔너리 리스트
            remove_outliers (bool): 이상치 제거 여부

        Returns:
            tuple: (월별 기준값, 월별 변동성 지표, 월별 분석 정보)
        """
        monthly_values = defaultdict(list)

        for pattern in usage_patterns:
            if isinstance(pattern, dict):
                for month, value in pattern.items():
                    try:
                        monthly_values[month].append(float(value))
                    except (ValueError, TypeError):
                        continue

        # 각 월별 최적화된 통계량 계산
        monthly_references = {}
        monthly_variabilities = {}
        monthly_analysis = {}

        for month, values in monthly_values.items():
            if values:
                # 이상치 제거 옵션 적용
                if remove_outliers and len(values) >= 4:
                    original_count = len(values)
                    values = self.remove_outliers_iqr(values)
                    if len(values) < original_count:
                        print(
                            f"      🔧 {month}: 이상치 제거 후 {len(values)}개 데이터 사용 (원래 {original_count}개)"
                        )

                # 분포 분석
                analysis = self.analyze_distribution(values)
                monthly_analysis[month] = analysis

                # 최적화된 기준값 계산
                if analysis["method"] == "trimmed_mean":
                    reference_val = self.calculate_trimmed_mean(values)
                    variability = np.std(values)  # 표준편차
                    method_label = "절삭평균"
                elif analysis["method"] == "median":
                    reference_val = np.median(values)
                    # IQR 사용
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    variability = q3 - q1
                    method_label = "중앙값"
                elif analysis["method"] == "mad_adjusted":
                    reference_val = self.calculate_mad_adjusted_center(values)
                    # MAD 사용
                    median_val = np.median(values)
                    variability = np.median(np.abs(values - median_val))
                    method_label = "MAD조정"
                else:  # fallback
                    reference_val = np.median(values)
                    variability = np.std(values)
                    method_label = "중앙값(기본)"

                monthly_references[month] = round(reference_val, 2)
                monthly_variabilities[month] = round(variability, 2)

                # 상세 분석 정보 출력
                print(
                    f"      📊 {month}: {method_label} = {reference_val:.2f} (±{variability:.2f})"
                )
                print(f"         - 분석근거: {analysis['reason']}")
                print(
                    f"         - 정규성 p-value: {analysis['normality_p']:.4f}"
                    if analysis["normality_p"]
                    else ""
                )
                print(
                    f"         - 치우침: {analysis['skewness']:.3f}"
                    if analysis["skewness"]
                    else ""
                )
                print(
                    f"         - MAD 비율: {analysis['mad_ratio']:.3f}"
                    if analysis["mad_ratio"]
                    else ""
                )
                print(f"         - 데이터 개수: {len(values)}개")

        return monthly_references, monthly_variabilities, monthly_analysis

    def calculate_monthly_stats_with_outlier_removal(
        self, usage_patterns, remove_outliers=False
    ):
        """
        적응형 통계 계산을 사용하도록 업데이트된 메서드
        (기존 호환성을 위해 유지)

        Args:
            usage_patterns (list): 사용량 패턴 딕셔너리 리스트
            remove_outliers (bool): 이상치 제거 여부

        Returns:
            tuple: (월별 기준값, 월별 변동성 지표)
        """
        references, variabilities, _ = self.calculate_adaptive_monthly_stats(
            usage_patterns, remove_outliers
        )
        return references, variabilities

    def calculate_monthly_stats(self, usage_patterns):
        """
        여러 사용량 패턴의 월별 통계를 계산

        Args:
            usage_patterns (list): 사용량 패턴 딕셔너리 리스트

        Returns:
            tuple: (월별 기준값, 월별 변동성 지표)
        """
        return self.calculate_monthly_stats_with_outlier_removal(
            usage_patterns, remove_outliers=False
        )

    def check_group_combinations(self, group_columns):
        """
        그룹화할 컬럼들의 고유한 조합 개수와 내용을 확인

        Args:
            group_columns (list): 그룹화할 컬럼명 리스트
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return

        # 고유한 조합 확인
        unique_combinations = self.df[group_columns].drop_duplicates()
        combination_count = len(unique_combinations)

        print(f"\n📊 그룹화 조합 분석:")
        print(f"   - 그룹화 기준: {', '.join(group_columns)}")
        print(f"   - 총 고유한 조합 개수: {combination_count}개")

        # 각 컬럼별 고유값 개수도 표시
        for col in group_columns:
            unique_values = self.df[col].nunique()
            print(f"   - '{col}' 고유값 개수: {unique_values}개")

        # 조합이 너무 많지 않으면 일부 예시 표시
        if combination_count <= 20:
            print(f"\n📋 모든 조합:")
            for idx, (_, row) in enumerate(unique_combinations.iterrows(), 1):
                combo_str = " | ".join([f"{col}: {row[col]}" for col in group_columns])
                # 각 조합별 데이터 개수도 표시
                count = len(
                    self.df[(self.df[group_columns] == row[group_columns]).all(axis=1)]
                )
                print(f"   [{idx:2d}] {combo_str} ({count}개 데이터)")
        else:
            print(f"\n📋 조합 예시 (처음 10개):")
            for idx, (_, row) in enumerate(unique_combinations.head(10).iterrows(), 1):
                combo_str = " | ".join([f"{col}: {row[col]}" for col in group_columns])
                count = len(
                    self.df[(self.df[group_columns] == row[group_columns]).all(axis=1)]
                )
                print(f"   [{idx:2d}] {combo_str} ({count}개 데이터)")
            print(f"   ... (총 {combination_count}개 조합)")

        return combination_count

    def create_heat_group_column(self):
        """
        열량 컬럼을 기반으로 열량 범위 그룹을 생성하고,
        '열량범위_그룹_용도' 형태의 새로운 컬럼을 추가
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return False

        # 필요한 컬럼들이 존재하는지 확인
        required_columns = ["열량", "그룹", "용도"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            print(f"필요한 컬럼이 없습니다: {missing_columns}")
            return False

        # 열량 범위 그룹 생성
        self.df["열량범위"] = self.df["열량"].apply(
            lambda x: get_heat_input_gt(x) if pd.notna(x) else "Unknown"
        )

        # 열량범위_그룹_용도 조합 컬럼 생성
        self.df["열량범위_그룹_용도"] = (
            self.df["열량범위"].astype(str)
            + "_"
            + self.df["그룹"].astype(str)
            + "_"
            + self.df["용도"].astype(str)
        )

        print(f"✅ 열량 범위 그룹화 완료")
        print(f"   - 생성된 열량 범위: {self.df['열량범위'].value_counts().to_dict()}")

        return True

    # 용도별 평균!!!!

    def group_and_calculate(self, group_columns=["그룹", "용도"]):
        """
        지정된 컬럼들을 기준으로 그룹화하고 적응형 통계 계산

        Args:
            group_columns (list): 그룹화할 컬럼명 리스트

        Returns:
            pd.DataFrame: 그룹화된 결과 데이터프레임
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return None

        # 필요한 컬럼들이 존재하는지 확인
        required_columns = group_columns + ["사용량_패턴"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            print(f"필요한 컬럼이 없습니다: {missing_columns}")
            return None

        # 그룹별로 데이터 처리
        grouped_results = []

        for group_value, group_df in self.df.groupby(group_columns):
            # 다중 컬럼 그룹화의 경우 group_value가 튜플이므로 처리
            if isinstance(group_value, tuple):
                group_info = dict(zip(group_columns, group_value))
            else:
                group_info = {group_columns[0]: group_value}

            print(f"\n🔍 그룹 분석 중: {group_info}")

            # 사용량 패턴 파싱 및 적응형 통계 계산
            usage_patterns = []
            for pattern in group_df["사용량_패턴"]:
                parsed_pattern = self.parse_usage_pattern(pattern)
                if parsed_pattern:
                    usage_patterns.append(parsed_pattern)

            # 월별 적응형 기준값과 변동성 지표 계산
            monthly_references, monthly_variabilities, monthly_analysis = (
                self.calculate_adaptive_monthly_stats(usage_patterns)
            )

            # 결과 저장 (그룹 정보와 통계 정보 포함)
            result_dict = group_info.copy()
            result_dict.update(
                {
                    "사용량 패턴 기준값": monthly_references,
                    "사용량 패턴 변동성": monthly_variabilities,
                    "분포 분석 정보": monthly_analysis,
                    "데이터 개수": len(group_df),
                }
            )
            grouped_results.append(result_dict)

        # 데이터프레임으로 변환
        self.grouped_data = pd.DataFrame(grouped_results)

        print(f"\n✅ 그룹화 완료: {len(self.grouped_data)}개 그룹")
        print(f"그룹화 기준: {', '.join(group_columns)}")
        return self.grouped_data

    def group_and_calculate_with_heat(self, group_columns=["열량범위", "그룹", "용도"]):
        """
        열량 범위를 포함한 지정된 컬럼들을 기준으로 그룹화하고 적응형 통계 계산

        Args:
            group_columns (list): 그룹화할 컬럼명 리스트

        Returns:
            pd.DataFrame: 그룹화된 결과 데이터프레임
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return None

        # 열량 범위 컬럼 생성
        if not self.create_heat_group_column():
            return None

        # 필요한 컬럼들이 존재하는지 확인
        required_columns = group_columns + ["사용량_패턴"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            print(f"필요한 컬럼이 없습니다: {missing_columns}")
            return None

        # 그룹별로 데이터 처리
        grouped_results = []

        for group_value, group_df in self.df.groupby(group_columns):
            # 다중 컬럼 그룹화의 경우 group_value가 튜플이므로 처리
            if isinstance(group_value, tuple):
                group_info = dict(zip(group_columns, group_value))
            else:
                group_info = {group_columns[0]: group_value}

            print(f"\n🔍 그룹 분석 중: {group_info}")

            # 사용량 패턴 파싱 및 적응형 통계 계산
            usage_patterns = []
            for pattern in group_df["사용량_패턴"]:
                parsed_pattern = self.parse_usage_pattern(pattern)
                if parsed_pattern:
                    usage_patterns.append(parsed_pattern)

            # 월별 적응형 기준값과 변동성 지표 계산
            monthly_references, monthly_variabilities, monthly_analysis = (
                self.calculate_adaptive_monthly_stats(usage_patterns)
            )

            # 결과 저장 (그룹 정보와 통계 정보 포함)
            result_dict = group_info.copy()
            result_dict.update(
                {
                    "사용량 패턴 기준값": monthly_references,
                    "사용량 패턴 변동성": monthly_variabilities,
                    "분포 분석 정보": monthly_analysis,
                    "데이터 개수": len(group_df),
                }
            )
            grouped_results.append(result_dict)

        # 데이터프레임으로 변환
        self.grouped_data = pd.DataFrame(grouped_results)

        print(f"\n✅ 그룹화 완료: {len(self.grouped_data)}개 그룹")
        print(f"그룹화 기준: {', '.join(group_columns)}")
        return self.grouped_data

    def display_results(self):
        """
        그룹화 결과를 화면에 출력 (적응형 통계 지원)
        """
        if self.grouped_data is None:
            print("그룹화된 데이터가 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f"                    적응형 통계 기반 그룹화 결과")
        print(f"{'='*80}")

        for idx, (_, row) in enumerate(self.grouped_data.iterrows(), 1):
            # 그룹 정보 추출 (통계 관련 컬럼 제외)
            group_info = {}
            for col in self.grouped_data.columns:
                if col not in [
                    "사용량 패턴 기준값",
                    "사용량 패턴 변동성",
                    "분포 분석 정보",
                    "데이터 개수",
                ]:
                    group_info[col] = row[col]

            print(f"\n[{idx}] ", end="")
            group_parts = [f"{k}: {v}" for k, v in group_info.items()]
            print(" | ".join(group_parts))

            # 데이터 개수 정보 추가
            if "데이터 개수" in row:
                print(f"     📊 데이터 개수: {row['데이터 개수']}개")

            print("-" * 60)

            monthly_reference_data = row["사용량 패턴 기준값"]
            monthly_variability_data = row["사용량 패턴 변동성"]
            monthly_analysis_data = row.get("분포 분석 정보", {})

            if isinstance(monthly_reference_data, dict) and isinstance(
                monthly_variability_data, dict
            ):
                print(f"📈 월별 적응형 기준값 및 변동성:")

                # 분기별로 나누어 출력
                quarters = [
                    ["1월", "2월", "3월"],  # 1분기
                    ["4월", "5월", "6월"],  # 2분기
                    ["7월", "8월", "9월"],  # 3분기
                    ["10월", "11월", "12월"],  # 4분기
                ]

                for quarter_idx, quarter in enumerate(quarters, 1):
                    quarter_data = []
                    for month in quarter:
                        ref_val = monthly_reference_data.get(month, 0)
                        var_val = monthly_variability_data.get(month, 0)

                        # 분석 정보에서 사용된 방법 확인
                        method_info = monthly_analysis_data.get(month, {})
                        method = method_info.get("method", "unknown")

                        # 방법별 표시 기호
                        method_symbol = {
                            "trimmed_mean": "🔹",  # 절삭평균
                            "median": "🔸",  # 중앙값
                            "mad_adjusted": "🔶",  # MAD조정
                        }.get(method, "⚪")

                        quarter_data.append(
                            f"{method_symbol}{month}: {ref_val:>7.2f}(±{var_val:>6.2f})"
                        )

                    print(f"   {quarter_idx}분기: {' | '.join(quarter_data)}")

                # 범례 표시
                print(f"\n   📋 범례: 🔹절삭평균 🔸중앙값 🔶MAD조정")

                # 분포 분석 요약 표시
                if monthly_analysis_data:
                    print(f"\n   🔍 분포 분석 요약:")
                    method_counts = {}
                    for month_info in monthly_analysis_data.values():
                        method = month_info.get("method", "unknown")
                        reason = month_info.get("reason", "unknown")
                        method_counts[f"{method} ({reason})"] = (
                            method_counts.get(f"{method} ({reason})", 0) + 1
                        )

                    for method_reason, count in method_counts.items():
                        print(f"      - {method_reason}: {count}개월")

        print(f"\n{'='*80}")
        print(f"💡 적응형 통계 설명:")
        print(f"   - 절삭평균: 정규분포에 가까운 데이터에 사용 (상하위 10% 제외)")
        print(f"   - 중앙값: 치우친 분포 데이터에 사용 (변동성=IQR)")
        print(f"   - MAD조정: 변동성이 큰 데이터에 사용 (MAD 기반 robust 중심값)")
        print(f"{'='*80}")


def convert_df_to_dict(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, list):
        return [convert_df_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_df_to_dict(v) for k, v in obj.items()}
    else:
        return obj


# 용도 별 그룹화
def main(input_file, output_file, group_columns, gt_json_path):
    """메인 실행 함수"""
    # 프로세서 초기화
    processor = ExcelGroupProcessor(input_file)

    # 데이터 로드
    if not processor.load_data():
        return

    # 그룹화 및 평균/표준편차 계산
    result = processor.group_and_calculate(group_columns=group_columns)
    results_serializable = convert_df_to_dict(result)
    # JSON 파일로 저장

    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=4)

    print(f"JSON 파일로 저장 완료: {gt_json_path}")

    return processor


# 열량 범위를 고려한 그룹화
def main_with_heat(input_file, output_file, group_columns, gt_json_path):
    """열량 범위를 고려한 메인 실행 함수"""
    # 프로세서 초기화
    processor = ExcelGroupProcessor(input_file)

    # 데이터 로드
    if not processor.load_data():
        return

    # 열량 범위를 고려한 그룹화 및 평균/표준편차 계산
    result = processor.group_and_calculate_with_heat(group_columns=group_columns)
    results_serializable = convert_df_to_dict(result)

    # JSON 파일로 저장
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=4)

    print(f"JSON 파일로 저장 완료: {gt_json_path}")

    return processor


if __name__ == "__main__":
    print("=" * 80)
    print("          Excel 데이터 그룹화 및 통계 분석 프로그램")
    print("=" * 80)

    # 기본 설정
    input_file = "./group_biz_with_12.xlsx"

    # 열량 범위를 고려한 그룹화 실행
    print("\n🔥 열량 범위를 고려한 그룹과 용도별 그룹화 실행...")
    output_file_heat = "./group_biz_with_usage_heat.xlsx"
    gt_json_path_heat = "./group_biz_with_usage_heat_optimize.json"
    group_columns_heat = ["열량범위", "그룹", "용도"]  # 열량범위, 그룹, 용도로 그룹화
    processor_heat = main_with_heat(
        input_file, output_file_heat, group_columns_heat, gt_json_path_heat
    )

    if processor_heat:
        print("✅ 열량 범위를 고려한 그룹과 용도별 그룹화 완료")
        processor_heat.display_results()

    print("\n" + "=" * 80)
    print("           분석이 완료되었습니다! 🎉")
    print("=" * 80)
