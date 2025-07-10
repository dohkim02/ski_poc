import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from utils import MonthlyAverageAnalyzer
import os
import sys
from utils import get_exel_with_biz_lst

data_path = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(data_path)

# 1️⃣ 데이터 로딩
file_path = "../data2_whole.xlsx"


# 필요한 컬럼 데이터만 뽑기
def preprocess_excel(file_path, output_path):
    # 1. 엑셀 파일 읽기
    df = pd.read_excel(file_path)

    # 2. 기본 컬럼 정의
    base_cols = ["구분", "업태", "업종", "용도", "등급"]

    # 3. 시간 컬럼 판별 (datetime 형식)
    time_cols = [col for col in df.columns if isinstance(col, datetime)]

    # 4. 업태 또는 업종 결측 제거
    df = df.dropna(subset=["업태", "업종"])

    # 5. 등급 컬럼 결측 → 0
    fill_zero_cols = ["등급"] + time_cols
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
    df_result = df[base_cols + ["사용량_패턴", "3년치 데이터"]]
    df_result.to_excel(output_path, index=False)

    print(f"✅ 처리 완료: {output_path}")
    return output_path


# 클러터링 할 청크 갯수 정하기
def chunk_list(data_list, chunk_size=10):
    return [data_list[i : i + chunk_size] for i in range(0, len(data_list), chunk_size)]


# 청크 데이터 쓰기
def write_chunk_line(file_path, chunk, lock):
    line = json.dumps(chunk, ensure_ascii=False)
    # 스레드가 동시에 파일에 쓰지 않도록 락을 사용!
    with lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# 업태 업종 뽑고, 저장
def get_category_list(file_path, output_file="./category_list.txt", chunk_size=10):
    import threading

    # 1️⃣ 엑셀 읽기
    df = pd.read_excel(file_path)

    # 2️⃣ '업태'와 '업종' 컬럼만 선택
    df_selected = df[["업태", "업종"]]

    # 3️⃣ JSON 변환
    data_list = json.loads(df_selected.to_json(orient="records", force_ascii=False))

    # 3️⃣ 10개씩 chunk로 나누기
    chunked_data = chunk_list(data_list, chunk_size)

    # 4️⃣ (쓰기 전에 기존 파일 삭제)
    with open(output_file, "w", encoding="utf-8") as f:
        pass  # 빈 파일로 초기화

    # 5️⃣ 멀티스레딩으로 각 청크를 파일에 한 줄씩 기록
    lock = threading.Lock()
    with ThreadPoolExecutor() as executor:
        for chunk in chunked_data:
            executor.submit(write_chunk_line, output_file, chunk, lock)

    return chunked_data


# output_path = "./preprocessed.xlsx"
# preprocess_excel(file_path, output_path)
# get_txt = get_category_list(output_path)


# # 사용 예시
# get_exel_with_biz_lst(
#     txt_path="./clustering_result.txt",
#     xlsx_path="./preprocessed.xlsx",
#     output_path="./group_biz_with_12.xlsx",
# )
# # 결과 확인
# print(df_result.head())
# print(f"\n전처리 후 데이터 개수: {len(df_result)}")
# print(f"결측치 확인:\n{df_result.isnull().sum()}")
