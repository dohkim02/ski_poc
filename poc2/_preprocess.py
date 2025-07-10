import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from utils import MonthlyAverageAnalyzer, clean_column_names, categorize_pressure


# 필요한 컬럼 데이터만 뽑기
def preprocess_excel(file_path, output_path):
    # 1. 엑셀 파일 읽기
    df = pd.read_excel(file_path)
    df = clean_column_names(df)

    # 2. 기본 컬럼 정의
    base_cols = ["구분", "업태", "업종", "용도", "등급", "압력"]

    # 3. 시간 컬럼 판별 (datetime 형식)
    time_cols = [col for col in df.columns if isinstance(col, datetime)]

    # 4. 업태 또는 업종 결측 제거
    df = df.dropna(subset=["업태", "업종"])

    # 5. 등급 컬럼 결측 → 0
    fill_zero_cols = ["등급", "압력"] + time_cols
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # 6. 3년치 데이터를 yearly_data 형태로 변환하는 함수
    def get_yearly_data(row):
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

    # 7. 3년치 데이터 컬럼 생성
    df["3년치 데이터"] = df.apply(get_yearly_data, axis=1)
    df["압력_그룹"] = df["압력"].apply(categorize_pressure)

    # 8. 사용량_패턴 생성 (월별 평균값 딕셔너리)
    def get_monthly_avg(row):
        periods = [col.strftime("%y.%m") for col in time_cols]
        values = [row[col] for col in time_cols]
        analyzer = MonthlyAverageAnalyzer()
        analyzer.load_data(periods, values)
        monthly = analyzer.calculate_monthly_averages()

        # {'1월': avg1, '2월': avg2, ...} 형태로 저장
        return dict(zip(monthly["month_name"], monthly["average"]))

    df["사용량_패턴"] = df.apply(get_monthly_avg, axis=1)

    # 9. 저장할 결과만 추출 (3년치 데이터 컬럼 포함)
    df_result = df[base_cols + ["압력_그룹", "사용량_패턴", "3년치 데이터"]]
    df_result.to_excel(output_path, index=False)

    print(f"✅ 처리 완료: {output_path}")
    return output_path


def excel_to_txt(file_path, output_file="./preprocessed.txt"):
    # 엑셀 읽기
    df = pd.read_excel(file_path)
    # JSON 변환
    data_list = json.loads(df.to_json(orient="records", force_ascii=False))
    # 파일 초기화 (기존 파일 삭제 또는 빈 파일로 덮어쓰기)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_file


# 이 부분이 문제였습니다 - 모듈 import 시 실행되는 코드들을 제거합니다
if __name__ == "__main__":
    # 테스트용 코드는 여기서만 실행되도록 합니다
    file_path = "./data2_test.xlsx"
    output_path = "./preprocessed.xlsx"
    preprocess_excel(file_path, output_path)
    get_txt = excel_to_txt(output_path)
    pass
