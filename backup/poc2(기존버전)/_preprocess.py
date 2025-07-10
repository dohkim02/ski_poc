import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from utils import MonthlyAverageAnalyzer

# 1️⃣ 데이터 로딩


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
