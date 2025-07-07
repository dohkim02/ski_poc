import pandas as pd
import json
import ast
from collections import defaultdict
import numpy as np
import os
import sys

data_path = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(data_path)


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

    def calculate_monthly_stats_with_outlier_removal(
        self, usage_patterns, remove_outliers=False
    ):
        """
        이상치 제거 옵션을 포함한 월별 중앙값 계산

        Args:
            usage_patterns (list): 사용량 패턴 딕셔너리 리스트
            remove_outliers (bool): 이상치 제거 여부

        Returns:
            tuple: (월별 중앙값, 월별 IQR)
        """
        monthly_values = defaultdict(list)

        for pattern in usage_patterns:
            if isinstance(pattern, dict):
                for month, value in pattern.items():
                    try:
                        monthly_values[month].append(float(value))
                    except (ValueError, TypeError):
                        continue

        # 각 월별 중앙값과 IQR 계산
        monthly_medians = {}
        monthly_iqrs = {}

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

                # 중앙값 계산
                median_val = np.median(values)
                monthly_medians[month] = round(median_val, 2)

                # IQR 계산 (Q3 - Q1)
                if len(values) >= 2:
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr_val = q3 - q1
                    monthly_iqrs[month] = round(iqr_val, 2)

                    # 디버깅 정보 출력 (IQR이 중앙값의 50% 이상인 경우)
                    if iqr_val > median_val * 0.5:
                        print(f"      ⚠️  {month} 높은 변동성 감지:")
                        print(
                            f"         - 중앙값: {median_val:.2f}, IQR: {iqr_val:.2f}"
                        )
                        print(f"         - 데이터 개수: {len(values)}개")
                        print(
                            f"         - 최소값: {min(values):.2f}, 최대값: {max(values):.2f}"
                        )
                        print(f"         - Q1: {q1:.2f}, Q3: {q3:.2f}")
                else:
                    monthly_iqrs[month] = 0.0

        return monthly_medians, monthly_iqrs

    def calculate_monthly_stats(self, usage_patterns):
        """
        여러 사용량 패턴의 월별 중앙값과 IQR을 계산

        Args:
            usage_patterns (list): 사용량 패턴 딕셔너리 리스트

        Returns:
            tuple: (월별 중앙값, 월별 IQR)
        """
        return self.calculate_monthly_stats_with_outlier_removal(
            usage_patterns, remove_outliers=False
        )

    # 용도별 평균!!!!
    def group_and_calculate(self, group_column="용도"):
        """
        지정된 컬럼을 기준으로 그룹화하고 평균값과 표준편차 계산

        Args:
            group_column (str): 그룹화할 컬럼명

        Returns:
            pd.DataFrame: 그룹화된 결과 데이터프레임
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return None

        # 필요한 컬럼들이 존재하는지 확인
        required_columns = [group_column, "사용량_패턴"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            print(f"필요한 컬럼이 없습니다: {missing_columns}")
            return None

        # 그룹별로 데이터 처리
        grouped_results = []

        for group_value, group_df in self.df.groupby(group_column):

            # 사용량 패턴 파싱 및 평균, 표준편차 계산
            usage_patterns = []
            for pattern in group_df["사용량_패턴"]:
                parsed_pattern = self.parse_usage_pattern(pattern)
                if parsed_pattern:
                    usage_patterns.append(parsed_pattern)

            # 월별 평균 사용량과 표준편차 계산
            monthly_medians, monthly_iqrs = (
                self.calculate_monthly_stats_with_outlier_removal(usage_patterns)
            )

            # 결과 저장
            grouped_results.append(
                {
                    group_column: group_value,
                    "사용량 패턴 중앙값": monthly_medians,
                    "사용량 패턴 IQR": monthly_iqrs,
                }
            )

        # 데이터프레임으로 변환
        self.grouped_data = pd.DataFrame(grouped_results)

        print(f"그룹화 완료: {len(self.grouped_data)}개 그룹")
        return self.grouped_data

    # 보일러, 연소기 열량 별 그룹화 기준
    def categorize_capacity(self, value):
        """
        열량값을 범위별 카테고리로 분류

        Args:
            value (float): 열량값

        Returns:
            str: 카테고리명
        """
        if pd.isna(value):
            return "미분류"
        elif value == 0:
            return "0"
        elif value < 10000:
            return "1~9999"
        elif value < 50000:
            return "10000~49999"
        elif value < 100000:
            return "50000~99999"
        else:
            return "100000 이상"

    # 보일러, 연소기 열량 별 평균 계산
    def group_by_capacity_and_calculate(self, remove_outliers=False):
        """
        보일러 열량과 연소기 열량을 합쳐서 총 열량을 계산하고 범위별로 그룹화하여 월별 평균과 표준편차 계산

        Args:
            remove_outliers (bool): 이상치 제거 여부 (기본값: False)

        Returns:
            pd.DataFrame: 총 열량 기준으로 그룹화된 결과
        """
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return None

        # 필요한 컬럼들이 존재하는지 확인
        required_columns = ["보일러 열량", "연소기 열량", "사용량_패턴"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            print(f"필요한 컬럼이 없습니다: {missing_columns}")
            return None

        print(f"🔍 원본 데이터 개수: {len(self.df)}개")
        if remove_outliers:
            print("📊 이상치 제거 모드 활성화")

        # 보일러 열량과 연소기 열량을 합쳐서 총 열량 계산
        print("보일러 열량과 연소기 열량을 합쳐서 총 열량 계산 중...")

        # 원본 데이터 분석
        boiler_na_count = self.df["보일러 열량"].isna().sum()
        combustor_na_count = self.df["연소기 열량"].isna().sum()
        print(f"   - 보일러 열량 결측값: {boiler_na_count}개")
        print(f"   - 연소기 열량 결측값: {combustor_na_count}개")

        self.df["총열량"] = self.df["보일러 열량"].fillna(0) + self.df[
            "연소기 열량"
        ].fillna(0)

        print(f"   - 총 열량 계산 후 데이터 개수: {len(self.df)}개")

        # 총 열량 그룹화
        print("총 열량 기준 그룹화 중...")
        self.df["총열량_그룹"] = self.df["총열량"].apply(self.categorize_capacity)

        # 그룹별 개수 확인
        group_counts = self.df["총열량_그룹"].value_counts()
        print(f"   - 그룹별 데이터 개수:")
        for group, count in group_counts.items():
            print(f"     {group}: {count}개")

        # 총 열량 기준으로 통계 계산 (이상치 제거 옵션 전달)
        total_capacity_results = self._calculate_capacity_group_stats(
            "총열량_그룹", "총열량", remove_outliers=remove_outliers
        )

        return total_capacity_results

    def _calculate_capacity_group_stats(
        self, group_column, capacity_column, remove_outliers=False
    ):
        """
        용량별 그룹의 통계를 계산하는 내부 함수

        Args:
            group_column (str): 그룹 컬럼명
            capacity_column (str): 용량 컬럼명
            remove_outliers (bool): 이상치 제거 여부

        Returns:
            pd.DataFrame: 그룹화된 결과 데이터프레임
        """
        grouped_results = []
        total_processed = 0
        total_usage_patterns_parsed = 0

        for group_value, group_df in self.df.groupby(group_column):
            if len(group_df) == 0:
                continue

            print(f"   📊 {group_value} 그룹 처리 중: {len(group_df)}개 데이터")
            total_processed += len(group_df)

            # 해당 용량의 평균과 표준편차
            capacity_avg = group_df[capacity_column].mean()
            capacity_std = (
                group_df[capacity_column].std(ddof=0) if len(group_df) > 1 else 0.0
            )

            # 사용량 패턴 파싱 및 평균, 표준편차 계산
            usage_patterns = []
            parsing_success_count = 0

            for pattern in group_df["사용량_패턴"]:
                parsed_pattern = self.parse_usage_pattern(pattern)
                if parsed_pattern:
                    usage_patterns.append(parsed_pattern)
                    parsing_success_count += 1

            print(
                f"      - 사용량 패턴 파싱 성공: {parsing_success_count}/{len(group_df)}개"
            )
            total_usage_patterns_parsed += parsing_success_count

            # 월별 평균 사용량과 표준편차 계산 (이상치 제거 옵션 적용)
            monthly_medians, monthly_iqrs = (
                self.calculate_monthly_stats_with_outlier_removal(
                    usage_patterns, remove_outliers=remove_outliers
                )
            )

            # 결과 저장
            grouped_results.append(
                {
                    "열량": group_value,
                    "데이터_수": len(group_df),
                    "사용량_패턴_중앙값": monthly_medians,
                    "사용량_패턴_IQR": monthly_iqrs,
                }
            )

        # 데이터프레임으로 변환
        result_df = pd.DataFrame(grouped_results)

        print(f"🔍 처리 결과 요약:")
        print(f"   - 총 처리된 데이터: {total_processed}개")
        print(f"   - 사용량 패턴 파싱 성공: {total_usage_patterns_parsed}개")
        print(f"   - {group_column} 그룹화 완료: {len(result_df)}개 그룹")

        return result_df

    def display_capacity_group_results(self, results):
        """
        총 열량별 그룹화 결과를 화면에 출력

        Args:
            results (pd.DataFrame): group_by_capacity_and_calculate 함수의 결과
        """
        if results is None:
            print("그룹화된 데이터가 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f"                    총 열량 범위별 그룹화 결과")
        print(f"{'='*80}")

        for idx, (_, row) in enumerate(results.iterrows(), 1):
            group_name = row["열량"]
            data_count = row["데이터_수"]

            print(f"\n[{idx}] 열량: {group_name} (데이터 수: {data_count}개)")
            print("-" * 60)

            monthly_avg_data = row["사용량_패턴_중앙값"]
            monthly_std_data = row["사용량_패턴_IQR"]

            if isinstance(monthly_avg_data, dict) and isinstance(
                monthly_std_data, dict
            ):
                print(f"📈 월별 사용량 (중앙값, IQR):")

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
                        avg_val = monthly_avg_data.get(month, 0)
                        std_val = monthly_std_data.get(month, 0)
                        quarter_data.append(
                            f"{month}: {avg_val:>7.2f}(±{std_val:>6.2f})"
                        )

                    print(f"   {quarter_idx}분기: {' | '.join(quarter_data)}")

        print(f"\n{'='*80}")

    def save_capacity_groups_to_excel(self, results, output_path):
        """
        총 열량별 그룹화 결과를 엑셀 파일로 저장

        Args:
            results (pd.DataFrame): 총 열량 그룹화 결과
            output_path (str): 저장할 파일 경로

        Returns:
            bool: 저장 성공 여부
        """
        if results is None:
            print("그룹화된 데이터가 없습니다.")
            return False

        try:
            # 그룹화 결과를 간단한 형태로 저장
            expanded_data = []

            for _, row in results.iterrows():
                new_row = {
                    "열량": row["열량"],
                    "데이터_수": row["데이터_수"],
                    "사용량_패턴_median": row[
                        "사용량_패턴_중앙값"
                    ],  # 딕셔너리 형태로 저장 (중앙값)
                    "사용량_패턴_IQR": row[
                        "사용량_패턴_IQR"
                    ],  # 딕셔너리 형태로 저장 (IQR)
                }
                expanded_data.append(new_row)

            result_df = pd.DataFrame(expanded_data)

            # 엑셀 파일로 저장
            result_df.to_excel(output_path, index=False, engine="openpyxl")

            print(f"결과가 저장되었습니다: {output_path}")
            print(f"총 {len(result_df)}개 그룹의 데이터가 저장되었습니다.")
            print("📊 저장된 컬럼:")
            print("   - 열량: 총 열량 구간")
            print("   - 데이터_수: 해당 구간의 데이터 개수")
            print("   - 사용량_패턴_median: 월별 사용량 중앙값 (딕셔너리)")
            print("   - 사용량_패턴_IQR: 월별 사용량 IQR (딕셔너리)")
            return True

        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return False

    def display_results(self):
        """
        용도별 그룹화 결과를 화면에 출력
        """
        if self.grouped_data is None:
            print("그룹화된 데이터가 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f"                    용도별 그룹화 결과")
        print(f"{'='*80}")

        for idx, (_, row) in enumerate(self.grouped_data.iterrows(), 1):
            group_column_name = self.grouped_data.columns[0]
            group_name = row[group_column_name]

            print(f"\n[{idx}] {group_column_name}: {group_name}")
            print("-" * 60)

            monthly_avg_data = row["사용량 패턴 중앙값"]
            monthly_std_data = row["사용량 패턴 IQR"]

            if isinstance(monthly_avg_data, dict) and isinstance(
                monthly_std_data, dict
            ):
                print(f"📈 월별 사용량 (중앙값, IQR):")

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
                        avg_val = monthly_avg_data.get(month, 0)
                        std_val = monthly_std_data.get(month, 0)
                        quarter_data.append(
                            f"{month}: {avg_val:>7.2f}(±{std_val:>6.2f})"
                        )

                    print(f"   {quarter_idx}분기: {' | '.join(quarter_data)}")

        print(f"\n{'='*80}")


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
def main(input_file, output_file, group_name, gt_json_path):
    """메인 실행 함수"""
    # 프로세서 초기화
    processor = ExcelGroupProcessor(input_file)

    # 데이터 로드
    if not processor.load_data():
        return

    # 그룹화 및 평균/표준편차 계산
    result = processor.group_and_calculate(group_column=group_name)
    results_serializable = convert_df_to_dict(result)
    # JSON 파일로 저장

    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=4)

    print(f"JSON 파일로 저장 완료: {gt_json_path}")

    return processor


def main_capacity_grouping(input_file, output_file, remove_outliers=False):
    """
    총 열량별 그룹화 메인 실행 함수

    Args:
        input_file (str): 입력 엑셀 파일 경로
        output_file (str): 출력 엑셀 파일 경로
        remove_outliers (bool): 이상치 제거 여부 (기본값: False)
    """
    print("🔥 총 열량(보일러+연소기) 범위별 그룹화 시작...")
    if remove_outliers:
        print("📊 이상치 제거 모드로 실행")
    print("=" * 80)

    # 프로세서 초기화
    processor = ExcelGroupProcessor(input_file)

    # 데이터 로드
    if not processor.load_data():
        print("❌ 데이터 로드 실패")
        return None

    # 총 열량별 그룹화 실행 (이상치 제거 옵션 전달)
    capacity_results = processor.group_by_capacity_and_calculate(
        remove_outliers=remove_outliers
    )

    if capacity_results is None:
        print("❌ 총 열량별 그룹화 실패")
        return None

    # 결과 화면 출력
    processor.display_capacity_group_results(capacity_results)

    # 엑셀 파일로 저장
    if processor.save_capacity_groups_to_excel(capacity_results, output_file):
        print(f"✅ 총 열량별 그룹화 결과 저장 완료: {output_file}")
    else:
        print("❌ 파일 저장 실패")

    return processor, capacity_results


if __name__ == "__main__":
    print("=" * 80)
    print("          Excel 데이터 그룹화 및 통계 분석 프로그램")
    print("=" * 80)

    # 기본 설정
    input_file = "./data2_preprocessed.xlsx"

    # 1. 기존 용도별 그룹화 실행
    print("\n🏢 1단계: 용도별 그룹화 실행...")
    output_file = "./data2_biz_with_std.xlsx"
    gt_json_path = "./biz_group_gt.json"
    group_name = "그룹"
    processor = main(input_file, output_file, group_name, gt_json_path)

    if processor:
        print("✅ 용도별 그룹화 완료")
        processor.display_results()

    print("\n" + "=" * 80)

    # 2. 새로운 용량별 그룹화 실행
    print("\n🔥 2단계: 총 열량별 그룹화 실행...")
    capacity_output_file = "./capacity_groups_analysis.xlsx"
    processor_capacity, capacity_results = main_capacity_grouping(
        input_file, capacity_output_file, remove_outliers=False
    )

    if capacity_results is not None:
        print("✅ 총 열량별 그룹화 완료 (기본 모드)")

        # 추가 분석 정보 출력
        print(f"\n📊 분석 완료 요약:")
        print(f"   - 총 열량 그룹: {len(capacity_results)}개")
        print(f"   - 결과 파일: {capacity_output_file}")

    print("\n" + "=" * 80)

    # 3. 이상치 제거 모드로 다시 실행
    print("\n🔥 3단계: 총 열량별 그룹화 실행 (이상치 제거 모드)...")
    capacity_output_file_clean = "./capacity_groups_analysis_clean.xlsx"
    processor_capacity_clean, capacity_results_clean = main_capacity_grouping(
        input_file, capacity_output_file_clean, remove_outliers=True
    )

    if capacity_results_clean is not None:
        print("✅ 총 열량별 그룹화 완료 (이상치 제거 모드)")

        # 추가 분석 정보 출력
        print(f"\n📊 이상치 제거 모드 분석 완료 요약:")
        print(f"   - 총 열량 그룹: {len(capacity_results_clean)}개")
        print(f"   - 결과 파일: {capacity_output_file_clean}")
        print(f"   - 두 결과를 비교하여 표준편차 변화를 확인하세요!")

    print("\n" + "=" * 80)
    print("           모든 분석이 완료되었습니다! 🎉")
    print("=" * 80)
