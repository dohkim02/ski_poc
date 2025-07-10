import pandas as pd
from datetime import datetime, timedelta
import calendar
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import time


def calculate_meter_usage_by_selection(excel_file_path, selection, sheet_name=None):
    """
    사용자 선택에 따라 계량기 사용량을 계산하는 함수

    Parameters:
    excel_file_path (str or file-like object): 엑셀 파일 경로 또는 파일 객체
    selection (str): 사용자 선택 ('계량기 당월 사용량', '계량기 전월 사용량', '계량기 전년동월 사용량', '사용량 평균')
    sheet_name (str): 시트 이름 (None이면 첫 번째 시트 사용)

    Returns:
    dict: 계산된 사용량 데이터
    """

    # 엑셀 파일 읽기 (파일 경로 또는 파일 객체 모두 지원)
    try:
        if hasattr(excel_file_path, "read"):
            # 파일 객체인 경우 (streamlit uploaded_file)
            df_or_dict = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        else:
            # 파일 경로인 경우
            df_or_dict = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"엑셀 파일을 읽을 수 없습니다: {str(e)}")

    # sheet_name이 None인 경우 딕셔너리가 반환되므로 첫 번째 시트를 가져옴
    if isinstance(df_or_dict, dict):
        # 첫 번째 시트의 DataFrame을 가져옴
        df = list(df_or_dict.values())[0]
    else:
        df = df_or_dict

    # 현재 날짜 구하기
    current_date = datetime.now()

    # 선택에 따른 날짜 계산
    if selection == "계량기 당월 사용량":
        # 현재 월의 전월 1일
        if current_date.month == 1:
            target_date = datetime(current_date.year - 1, 12, 1)
        else:
            target_date = datetime(current_date.year, current_date.month - 1, 1)

        # 전월 사용량 날짜 (당월의 바로 전월 1일)
        if target_date.month == 1:
            previous_month_date = datetime(target_date.year - 1, 12, 1)
        else:
            previous_month_date = datetime(target_date.year, target_date.month - 1, 1)

        # 전년동월 사용량 날짜 (당월의 전년도 같은 월 1일)
        previous_year_date = datetime(target_date.year - 1, target_date.month, 1)

        print(f"현재 날짜: {current_date.strftime('%Y.%m.%d')}")
        print(f"계량기 당월 사용량 기준일: {target_date.strftime('%Y.%m.%d')}")
        print(f"계량기 전월 사용량 기준일: {previous_month_date.strftime('%Y.%m.%d')}")
        print(
            f"계량기 전년동월 사용량 기준일: {previous_year_date.strftime('%Y.%m.%d')}"
        )

        # 결과 계산
        result = get_usage_data(
            df, target_date, previous_month_date, previous_year_date
        )

    elif selection == "계량기 전월 사용량":
        # 현재 월의 전전월 1일
        if current_date.month <= 2:
            target_date = datetime(
                current_date.year - 1, 12 - (2 - current_date.month), 1
            )
        else:
            target_date = datetime(current_date.year, current_date.month - 2, 1)

        print(f"현재 날짜: {current_date.strftime('%Y.%m.%d')}")
        print(f"계량기 전월 사용량 기준일: {target_date.strftime('%Y.%m.%d')}")

        result = get_single_usage_data(df, target_date, "전월")

    elif selection == "계량기 전년동월 사용량":
        # 현재 월의 전년도 전월 1일
        if current_date.month == 1:
            target_date = datetime(current_date.year - 2, 12, 1)
        else:
            target_date = datetime(current_date.year - 1, current_date.month - 1, 1)

        print(f"현재 날짜: {current_date.strftime('%Y.%m.%d')}")
        print(f"계량기 전년동월 사용량 기준일: {target_date.strftime('%Y.%m.%d')}")

        result = get_single_usage_data(df, target_date, "전년동월")

    elif selection == "사용량 평균":
        print(f"현재 날짜: {current_date.strftime('%Y.%m.%d')}")
        print("사용량 평균을 엑셀에서 가져옵니다.")

        result = get_average_usage_data(df)

    else:
        raise ValueError(
            "잘못된 선택입니다. '계량기 당월 사용량', '계량기 전월 사용량', '계량기 전년동월 사용량', '사용량 평균' 중 하나를 선택하세요."
        )

    return result


def get_usage_data(df, current_date, previous_date, previous_year_date):
    """
    날짜별 사용량을 각 세대에 대해 추출
    """
    # 날짜 기준 문자열 변환 (두 가지 형식 모두 지원)
    current_str_dash = current_date.strftime("%Y-%m-%d")
    current_str_dot = current_date.strftime("%Y.%m.%d")
    previous_str_dash = previous_date.strftime("%Y-%m-%d")
    previous_str_dot = previous_date.strftime("%Y.%m.%d")
    previous_year_str_dash = previous_year_date.strftime("%Y-%m-%d")
    previous_year_str_dot = previous_year_date.strftime("%Y.%m.%d")

    # 날짜 컬럼 찾기 (통합된 방식)
    date_columns = find_date_columns_enhanced(df)

    # 날짜 컬럼을 문자열로 변환해 매핑 딕셔너리 만들기
    date_column_map = {}

    for col in date_columns:
        if isinstance(col, datetime):
            # datetime 컬럼인 경우
            dash_format = col.strftime("%Y-%m-%d")
            dot_format = col.strftime("%Y.%m.%d")
            date_column_map[dash_format] = col
            date_column_map[dot_format] = col
        else:
            # 문자열 컬럼인 경우, 샘플 값으로 형식 확인
            try:
                sample_val = df[col].dropna().iloc[0]
                if isinstance(sample_val, str):
                    # 문자열 날짜인 경우
                    if "." in sample_val:
                        date_column_map[sample_val] = col
                    elif "-" in sample_val:
                        date_column_map[sample_val] = col
                elif isinstance(sample_val, datetime):
                    # datetime 값인 경우
                    dash_format = sample_val.strftime("%Y-%m-%d")
                    dot_format = sample_val.strftime("%Y.%m.%d")
                    date_column_map[dash_format] = col
                    date_column_map[dot_format] = col
            except:
                continue

    # 두 가지 형식 모두 시도하여 컬럼 찾기
    current_col = date_column_map.get(current_str_dash) or date_column_map.get(
        current_str_dot
    )
    previous_col = date_column_map.get(previous_str_dash) or date_column_map.get(
        previous_str_dot
    )
    previous_year_col = date_column_map.get(
        previous_year_str_dash
    ) or date_column_map.get(previous_year_str_dot)

    # 딕셔너리 형태로 결과 반환 (app2.py 호환성을 위해)
    result = {
        "selection": "계량기 당월 사용량",
        "current_month_usage": (
            df[current_col].iloc[0]
            if current_col is not None and current_col in df.columns
            else None
        ),
        "previous_month_usage": (
            df[previous_col].iloc[0]
            if previous_col is not None and previous_col in df.columns
            else None
        ),
        "previous_year_usage": (
            df[previous_year_col].iloc[0]
            if previous_year_col is not None and previous_year_col in df.columns
            else None
        ),
        "average_usage": (
            df["사용량 평균"].iloc[0] if "사용량 평균" in df.columns else None
        ),
    }

    return result


def get_single_usage_data(df, target_date, usage_type):
    """
    단일 사용량 데이터를 가져오는 함수
    """
    # 날짜 컬럼 찾기
    date_columns = find_date_columns_enhanced(df)

    # 날짜 컬럼들을 datetime 형식으로 변환
    for col in date_columns:
        if not isinstance(col, datetime):
            # 두 가지 형식 모두 시도
            try:
                df[col] = pd.to_datetime(df[col], format="%Y.%m.%d", errors="coerce")
            except:
                try:
                    df[col] = pd.to_datetime(
                        df[col], format="%Y-%m-%d", errors="coerce"
                    )
                except:
                    pass

    # 사용량 컬럼 찾기
    usage_columns = [
        col for col in df.columns if "사용량" in str(col) and "평균" not in str(col)
    ]

    result = {"selection": f"계량기 {usage_type} 사용량", "usage_value": None}

    # 해당 날짜의 사용량 찾기
    for date_col in date_columns:
        try:
            if isinstance(date_col, datetime):
                # 컬럼명이 datetime인 경우
                if date_col.date() == target_date.date():
                    for usage_col in usage_columns:
                        if not df[usage_col].empty:
                            result["usage_value"] = df[usage_col].iloc[0]
                            break
                    break
            else:
                # 컬럼명이 문자열인 경우, 값으로 비교
                target_mask = df[date_col] == target_date
                if target_mask.any():
                    for usage_col in usage_columns:
                        if not df.loc[target_mask, usage_col].empty:
                            result["usage_value"] = df.loc[target_mask, usage_col].iloc[
                                0
                            ]
                            break
                    break
        except Exception as e:
            continue

    return result


def get_average_usage_data(df):
    """
    평균 사용량 데이터를 가져오는 함수
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "get_average_usage_data 함수에는 DataFrame만 전달해야 합니다. 현재 타입: {}".format(
                type(df)
            )
        )
    result = {"selection": "사용량 평균", "average_usage": None}

    # 정확히 '사용량 평균' 컬럼 찾기
    if "사용량 평균" in df.columns:
        result["average_usage"] = df["사용량 평균"].iloc[0]
    else:
        # 백업으로 '평균'이 포함된 컬럼 찾기
        average_columns = [col for col in df.columns if "평균" in str(col)]
        if average_columns:
            result["average_usage"] = df[average_columns[0]].iloc[0]

    return result


def find_date_columns_enhanced(df):
    """
    향상된 날짜 컬럼 찾기 함수 (datetime 타입과 문자열 날짜 모두 지원)
    """
    date_columns = []

    for col in df.columns:
        # 1. datetime 타입인 컬럼 확인
        if isinstance(col, datetime):
            date_columns.append(col)
            continue

        # 2. 컬럼 값들이 datetime 타입인지 확인
        if df[col].dtype.name.startswith("datetime"):
            date_columns.append(col)
            continue

        # 3. 문자열 형태의 날짜 컬럼 확인
        if df[col].dtype == "object":
            try:
                # 문자열을 날짜로 변환 시도
                sample_value = df[col].dropna().iloc[0]
                if isinstance(sample_value, str):
                    # 두 가지 형식 모두 시도
                    try:
                        pd.to_datetime(sample_value, format="%Y.%m.%d")
                        date_columns.append(col)
                        continue
                    except:
                        pass
                    try:
                        pd.to_datetime(sample_value, format="%Y-%m-%d")
                        date_columns.append(col)
                        continue
                    except:
                        pass
                elif isinstance(sample_value, datetime):
                    date_columns.append(col)
            except:
                continue

    return date_columns


def print_usage_result(result):
    """
    계산 결과를 보기 좋게 출력하는 함수
    """
    print(f"\n=== {result['selection']} 결과 ===")

    if result["selection"] == "계량기 당월 사용량":
        print(f"1. 계량기 당월 사용량: {result.get('current_month_usage', 'N/A')}")
        print(f"2. 계량기 전월 사용량: {result.get('previous_month_usage', 'N/A')}")
        print(f"3. 계량기 전년동월 사용량: {result.get('previous_year_usage', 'N/A')}")
        print(f"4. 사용량 평균: {result.get('average_usage', 'N/A')}")
    elif result["selection"] == "사용량 평균":
        print(f"사용량 평균: {result.get('average_usage', 'N/A')}")
    else:
        print(f"{result['selection']}: {result.get('usage_value', 'N/A')}")


# 사용 예시
if __name__ == "__main__":
    # 엑셀 파일 경로를 입력하세요
    excel_file_path = "./data/data2.xlsx"  # 실제 파일 경로로 변경

    # 사용자 선택 (아래 중 하나 선택)
    user_selections = [
        "계량기 당월 사용량",
        "계량기 전월 사용량",
        "계량기 전년동월 사용량",
        "사용량 평균",
    ]

    # 예시: 계량기 당월 사용량 선택으로 다시 변경
    selected_option = "계량기 당월 사용량"

    try:
        # 선택에 따른 계량기 사용량 계산
        result = calculate_meter_usage_by_selection(excel_file_path, selected_option)

        # 결과 출력
        print_usage_result(result)

    except FileNotFoundError:
        print("엑셀 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback

        traceback.print_exc()  # 상세한 오류 정보 출력


# 모든 선택 옵션 테스트
def test_all_selections(excel_file_path):
    """
    모든 선택 옵션을 테스트하는 함수
    """
    selections = [
        "계량기 당월 사용량",
        "계량기 전월 사용량",
        "계량기 전년동월 사용량",
        "사용량 평균",
    ]

    for selection in selections:
        print(f"\n{'='*50}")
        print(f"테스트 중: {selection}")
        print(f"{'='*50}")

        try:
            result = calculate_meter_usage_by_selection(excel_file_path, selection)
            print_usage_result(result)
        except Exception as e:
            print(f"오류 발생: {str(e)}")


# 결과 예시:
# === 계량기 당월 사용량 결과 ===
# 1. 계량기 당월 사용량: 150
# 2. 계량기 전월 사용량: 140
# 3. 계량기 전년동월 사용량: 145
# 4. 사용량 평균: 142


# 엑셀 파일 불러오기
df = pd.read_excel("./data/data2.xlsx")  # 예: "data.xlsx"

# 1. 컬럼 전체 출력
print("📌 컬럼 목록:")
print(df.columns.tolist())

# 2. 각 컬럼의 데이터 타입 확인
print("\n📌 컬럼별 데이터 타입:")
print(df.dtypes)


def process_single_row(
    row_data,
    selected_A,
    selected_B,
    op,
    threshold_value,
    drop_ratio,
    target_date,
    previous_date,
    previous_year_date,
):
    """
    단일 행에 대해 계량기 이상 징후를 분석하는 함수
    """
    try:
        row = row_data

        # 날짜별 사용량 추출
        usage_values = extract_usage_from_row(
            row, target_date, previous_date, previous_year_date
        )

        if not usage_values:
            return {"result": "데이터 없음", "details": {}}

        current = usage_values.get("current_month_usage")
        prev = usage_values.get("previous_month_usage")
        prev_year = usage_values.get("previous_year_usage")
        avg = usage_values.get("average_usage")

        # A, B 조건 값 매핑
        value_mapping = {
            "계량기 당월 사용량": current,
            "계량기 전월 사용량": prev,
            "계량기 전년동월 사용량": prev_year,
        }

        A_value = value_mapping.get(selected_A, 0)
        B_value = value_mapping.get(selected_B, 0)

        # 조건 확인
        cond_A = A_value is not None and A_value >= threshold_value
        cond_B = B_value is not None and B_value >= threshold_value
        match = (cond_A and cond_B) if op == "and" else (cond_A or cond_B)

        # 이상 징후 판단
        if not match:
            result = "조건 불충족"
        elif current == 0:
            result = "미사용세대"
        elif prev and prev > 0 and current / prev <= drop_ratio:
            result = "전월 대비 급감"
        elif prev_year and prev_year > 0 and current / prev_year <= drop_ratio:
            result = "전년동월 대비 급감"
        elif avg and avg > 0 and current / avg <= drop_ratio:
            result = "사용량 평균 대비 급감"
        else:
            result = "정상"

        return {
            "result": result,
            "details": {
                "계량기 당월 사용량": current,
                "계량기 전월 사용량": prev,
                "계량기 전년동월 사용량": prev_year,
                "사용량 평균": avg,
                "A_value": A_value,
                "B_value": B_value,
                "condition_match": match,
            },
        }

    except Exception as e:
        return {"result": f"처리 오류: {str(e)}", "details": {}}


def extract_usage_from_row(row, target_date, previous_date, previous_year_date):
    """
    행에서 날짜별 사용량을 추출하는 함수
    """
    try:
        # 날짜 문자열 생성
        current_str_dash = target_date.strftime("%Y-%m-%d")
        current_str_dot = target_date.strftime("%Y.%m.%d")
        previous_str_dash = previous_date.strftime("%Y-%m-%d")
        previous_str_dot = previous_date.strftime("%Y.%m.%d")
        previous_year_str_dash = previous_year_date.strftime("%Y-%m-%d")
        previous_year_str_dot = previous_year_date.strftime("%Y.%m.%d")

        # 가능한 모든 날짜 형식으로 컬럼 찾기
        possible_current_keys = [current_str_dash, current_str_dot]
        possible_previous_keys = [previous_str_dash, previous_str_dot]
        possible_prev_year_keys = [previous_year_str_dash, previous_year_str_dot]

        current_usage = None
        previous_usage = None
        prev_year_usage = None

        # 각 키로 값 찾기
        for key in possible_current_keys:
            if key in row and pd.notna(row[key]):
                current_usage = row[key]
                break

        for key in possible_previous_keys:
            if key in row and pd.notna(row[key]):
                previous_usage = row[key]
                break

        for key in possible_prev_year_keys:
            if key in row and pd.notna(row[key]):
                prev_year_usage = row[key]
                break

        # 평균 사용량
        avg_usage = row.get("사용량 평균")
        if pd.isna(avg_usage):
            avg_usage = None

        return {
            "current_month_usage": current_usage,
            "previous_month_usage": previous_usage,
            "previous_year_usage": prev_year_usage,
            "average_usage": avg_usage,
        }

    except Exception as e:
        return {}


def analyze_all_rows_parallel(
    excel_file_path,
    selected_A,
    selected_B,
    op,
    threshold_value,
    drop_ratio,
    max_workers=None,
    progress_callback=None,
):
    """
    모든 행에 대해 병렬로 계량기 이상 징후를 분석하는 함수
    """
    # 파일 읽기
    try:
        if hasattr(excel_file_path, "read"):
            df_or_dict = pd.read_excel(excel_file_path)
        else:
            df_or_dict = pd.read_excel(excel_file_path)
    except Exception as e:
        raise ValueError(f"엑셀 파일을 읽을 수 없습니다: {str(e)}")

    if isinstance(df_or_dict, dict):
        df = list(df_or_dict.values())[0]
    else:
        df = df_or_dict

    # 현재 날짜 기준으로 분석 날짜 계산
    current_date = datetime.now()

    # 현재 월의 전월 1일
    if current_date.month == 1:
        target_date = datetime(current_date.year - 1, 12, 1)
    else:
        target_date = datetime(current_date.year, current_date.month - 1, 1)

    # 전월 사용량 날짜 (당월의 바로 전월 1일)
    if target_date.month == 1:
        previous_month_date = datetime(target_date.year - 1, 12, 1)
    else:
        previous_month_date = datetime(target_date.year, target_date.month - 1, 1)

    # 전년동월 사용량 날짜 (당월의 전년도 같은 월 1일)
    previous_year_date = datetime(target_date.year - 1, target_date.month, 1)

    # 데이터를 딕셔너리 형태로 변환
    row_data_list = df.to_dict("records")
    total_rows = len(row_data_list)

    if max_workers is None:
        max_workers = min(os.cpu_count(), 8)  # 최대 8개 워커

    results = []
    completed_count = 0
    start_time = time.time()

    # 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 모든 행을 병렬로 처리
        future_to_index = {
            executor.submit(
                process_single_row,
                row,
                selected_A,
                selected_B,
                op,
                threshold_value,
                drop_ratio,
                target_date,
                previous_month_date,
                previous_year_date,
            ): i
            for i, row in enumerate(row_data_list)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
                completed_count += 1

                # 진행률 업데이트
                if progress_callback:
                    progress = completed_count / total_rows
                    elapsed_time = time.time() - start_time
                    rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                    progress_callback(progress, rate)

            except Exception as e:
                results.append((index, {"result": f"오류: {str(e)}", "details": {}}))
                completed_count += 1

    # 결과를 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])

    # 결과를 DataFrame에 추가
    result_data = [result[1] for result in results]
    df["이상징후_결과"] = [r["result"] for r in result_data]
    df["당월사용량"] = [r["details"].get("계량기 당월 사용량") for r in result_data]
    df["전월사용량"] = [r["details"].get("계량기 전월 사용량") for r in result_data]
    df["전년동월사용량"] = [
        r["details"].get("계량기 전년동월 사용량") for r in result_data
    ]
    df["평균사용량"] = [r["details"].get("사용량 평균") for r in result_data]
    df["조건만족여부"] = [r["details"].get("condition_match") for r in result_data]

    return df
