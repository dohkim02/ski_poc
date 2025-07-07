import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from utils import MonthlyAverageAnalyzer

# 1️⃣ 데이터 로딩
file_path = "./data2_test.xlsx"


# 필요한 컬럼 데이터만 뽑기
def preprocess_excel(file_path, output_path):
    # 1. 엑셀 파일 읽기
    df = pd.read_excel(file_path)

    # 2. 기본 컬럼 정의
    base_cols = [
        "구분",
        "업태",
        "업종",
        "용도",
        "보일러 대수",
        "보일러 열량",
        "연소기 대수",
        "연소기 열량",
        "열량",  # 새로운 열량 컬럼 추가
    ]

    # 3. 시간 컬럼 판별 (datetime 형식)
    time_cols = [col for col in df.columns if isinstance(col, datetime)]

    # 4. 업태 또는 업종 결측 제거
    df = df.dropna(subset=["업태", "업종"])

    # 5. 보일러, 연소기, 시간 컬럼 결측 → 0
    fill_zero_cols = [
        "보일러 대수",
        "보일러 열량",
        "연소기 대수",
        "연소기 열량",
    ] + time_cols
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # 6. 열량 컬럼 생성: (보일러 대수 × 보일러 열량) + (연소기 대수 × 연소기 열량)
    df["열량"] = (df["보일러 대수"] * df["보일러 열량"]) + (
        df["연소기 대수"] * df["연소기 열량"]
    )

    # 7. 열량이 10000 미만인 행 제거
    df = df[df["열량"] >= 10000]

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

    # 9. 저장할 결과만 추출
    df_result = df[base_cols + ["사용량_패턴"]]
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


output_path = "./preprocessed.xlsx"
# preprocess_excel(file_path, output_path)
get_txt = excel_to_txt(output_path)
