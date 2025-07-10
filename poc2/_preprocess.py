import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from utils import MonthlyAverageAnalyzer, clean_column_names, categorize_pressure


# 필요한 컬럼 데이터만 뽑기
def preprocess_excel(file_path, output_path):
    """
    Excel 파일 전처리 (Streamlit Cloud 호환)

    Args:
        file_path: 입력 Excel 파일 경로
        output_path: 출력 Excel 파일 경로
    """
    try:
        print(f"🔄 Starting preprocessing: {file_path}")

        # 1. 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        df = clean_column_names(df)

        print(f"📊 Loaded data: {len(df)} rows, {len(df.columns)} columns")

        # 2. 기본 컬럼 정의
        base_cols = ["구분", "업태", "업종", "용도", "등급", "압력"]

        # 3. 시간 컬럼 판별 (datetime 형식)
        time_cols = [col for col in df.columns if isinstance(col, datetime)]
        print(f"🕐 Found {len(time_cols)} time columns")

        # 4. 업태 또는 업종 결측 제거
        initial_len = len(df)
        df = df.dropna(subset=["업태", "업종"])
        removed_count = initial_len - len(df)
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} rows with missing 업태/업종")

        # 5. 등급 컬럼 결측 → 0
        fill_zero_cols = ["등급", "압력"] + time_cols
        df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

        # 6. 3년치 데이터를 yearly_data 형태로 변환하는 함수
        def get_yearly_data(row):
            try:
                # 시간 컬럼들을 yy.mm 형태로 변환하고 값들을 가져옴
                time_data = {col.strftime("%y.%m"): row[col] for col in time_cols}

                # yearly_data 형태로 변환 (년도별로 그룹화)
                yearly_data = {}
                for key, value in time_data.items():
                    year, month = key.split(".")
                    if year not in yearly_data:
                        yearly_data[year] = {}
                    yearly_data[year][month] = value

                return yearly_data
            except Exception as e:
                print(f"⚠️  Warning in get_yearly_data: {str(e)}")
                return {}

        # 7. 3년치 데이터 컬럼 생성
        print("🔄 Creating 3년치 데이터 column...")
        df["3년치 데이터"] = df.apply(get_yearly_data, axis=1)
        df["압력_그룹"] = df["압력"].apply(categorize_pressure)

        # 8. 사용량_패턴 생성 (월별 평균값 딕셔너리)
        def get_monthly_avg(row):
            try:
                periods = [col.strftime("%y.%m") for col in time_cols]
                values = [row[col] for col in time_cols]
                analyzer = MonthlyAverageAnalyzer()
                analyzer.load_data(periods, values)
                monthly = analyzer.calculate_monthly_averages()

                # {'1월': avg1, '2월': avg2, ...} 형태로 저장
                return dict(zip(monthly["month_name"], monthly["average"]))
            except Exception as e:
                print(f"⚠️  Warning in get_monthly_avg: {str(e)}")
                return {}

        print("🔄 Creating 사용량_패턴 column...")
        df["사용량_패턴"] = df.apply(get_monthly_avg, axis=1)

        # 9. 저장할 결과만 추출 (3년치 데이터 컬럼 포함)
        df_result = df[base_cols + ["압력_그룹", "사용량_패턴", "3년치 데이터"]]

        # 10. 파일 저장
        df_result.to_excel(output_path, index=False)

        print(f"✅ Preprocessing completed: {output_path}")
        print(
            f"📊 Final result: {len(df_result)} rows, {len(df_result.columns)} columns"
        )

        return output_path

    except Exception as e:
        print(f"❌ Error in preprocess_excel: {str(e)}")
        import traceback

        print(f"📄 Traceback: {traceback.format_exc()}")

        # 에러 발생 시에도 출력 경로 반환
        return output_path


def excel_to_txt(file_path, output_file=None):
    """
    Excel 파일을 텍스트 파일로 변환 (Streamlit Cloud 호환)

    Args:
        file_path: 입력 Excel 파일 경로
        output_file: 출력 파일 경로 (None이면 임시 파일 생성)
    """
    try:
        # 출력 파일이 지정되지 않으면 임시 파일 생성
        if output_file is None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp_file:
                output_file = tmp_file.name

        # 엑셀 읽기
        df = pd.read_excel(file_path)

        # 데이터가 있는지 확인
        if df.empty:
            print("⚠️  Warning: Excel file is empty")
            return output_file

        print(f"✅ Excel file loaded successfully. Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")

        # JSON 변환
        data_list = json.loads(df.to_json(orient="records", force_ascii=False))

        print(f"🔄 Converted to JSON. Number of records: {len(data_list)}")

        # 파일 쓰기 (명시적으로 텍스트 모드로)
        with open(output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data_list):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"💾 TXT file saved successfully: {output_file}")

        # 저장된 파일 검증
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"✅ Verification: {len(lines)} lines written to file")
            if len(lines) > 0:
                # 첫 번째 라인 확인
                first_item = json.loads(lines[0].strip())
                print(f"🔍 First item keys: {list(first_item.keys())}")

        return output_file

    except Exception as e:
        print(f"❌ Error in excel_to_txt: {str(e)}")
        import traceback

        print(f"📄 Traceback: {traceback.format_exc()}")

        # 에러가 발생해도 출력 파일 경로는 반환 (빈 파일이라도)
        if output_file is None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp_file:
                output_file = tmp_file.name
        return output_file


# 이 부분이 문제였습니다 - 모듈 import 시 실행되는 코드들을 제거합니다
if __name__ == "__main__":
    # 테스트용 코드는 여기서만 실행되도록 합니다
    file_path = "./data2_test.xlsx"
    output_path = "./preprocessed.xlsx"
    preprocess_excel(file_path, output_path)
    get_txt = excel_to_txt(output_path)
    pass
