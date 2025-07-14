import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor


# 데이터 중 결측이 있는 행 제거
def drop_na(file_path):
    df = pd.read_excel(file_path, sheet_name="학습데이터2(연소기정보)")
    print("처음 행 개수:", len(df))

    # 결측값이 있는 행을 제거합니다.
    df = df.dropna()
    print("결측값 제거 후 행 개수:", len(df))

    # '구분' 컬럼을 숫자형으로 변환 후, 오름차순 정렬
    df["구분"] = df["구분"].astype(int)
    df = df.sort_values(by="구분", ascending=True)
    print("구분 컬럼을 숫자로 변환하고 정렬되었습니다.")

    # 결과를 저장합니다.
    output_path = "../data/학습데이터2(연소기정보).xlsx"
    df.to_excel(output_path, index=False)

    print("결측값이 있는 행이 삭제되고, 구분 기준으로 오름차순 정렬되었습니다.")
    return output_path


# 제거된 데이터가 있는 엑셀을 기준으로 다른 엑셀도 구분 숫자 맞춤
def data_split(file_path, output_path):
    # 두 개의 엑셀 파일 불러오기
    df1 = pd.read_excel(output_path)
    df2 = pd.read_excel(file_path, sheet_name="학습데이터1(계량기자원정보)")

    # '구분' 컬럼을 숫자형으로 변환
    df1["구분"] = df1["구분"].astype(int)
    df2["구분"] = df2["구분"].astype(int)

    # 기준이 되는 '구분' 값 리스트 (file1의 '구분' 기준)
    criteria_list = df1["구분"].unique()

    # file2의 '구분' 값이 기준 리스트에 있는 데이터만 추출
    matched_df = df2[df2["구분"].isin(criteria_list)]

    # 결과를 엑셀로 저장
    new_output_path = "../data/학습데이터1(계량기자원정보).xlsx"
    matched_df.to_excel(new_output_path, index=False)

    print(
        "두 파일의 '구분' 컬럼을 숫자형으로 변환 후, 일치하는 데이터가 저장되었습니다."
    )
    return new_output_path


# 데이터 머지
def data_merge(file_path1, file_path2):
    df1 = pd.read_excel(file_path1)
    df2 = pd.read_excel(file_path2)

    # 1️⃣ merge: '구분'으로 왼쪽 조인
    merged = pd.merge(df1, df2, on="구분", how="left")

    # 2️⃣ groupby로 각 '구분'별로 연소기정보를 리스트로 묶어줌
    grouped_list = []
    for key, group in merged.groupby(
        ["구분", "고지형식", "세대유형", "사용(측정)압력", "등급"]
    ):
        record = {
            "구분": key[0],
            "고지형식": key[1],
            "세대유형": key[2],
            "사용(측정)압력": key[3],
            "등급": key[4],
            "연소기정보": group[["연소기명", "수량", "열량", "산업용 여부"]].to_dict(
                orient="records"
            ),
        }
        grouped_list.append(record)

    # 최종 DataFrame으로 변환
    grouped_df = pd.DataFrame(grouped_list)
    # data_merge 함수 마지막에 추가
    output_file = "./merged_data.xlsx"
    grouped_df.to_excel(output_file, index=False)
    print(f"병합된 데이터가 {output_file}에 저장되었습니다.")
    return output_file


def main():
    file_path = "../data/data1.xlsx"
    output_path = drop_na(file_path)
    new_output_path = data_split(file_path, output_path)

    merge_path = data_merge(output_path, new_output_path)

    return merge_path
