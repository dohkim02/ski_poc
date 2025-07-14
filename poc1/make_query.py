import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def create_single_query(row):
    """단일 행에 대한 쿼리를 생성하는 함수"""
    try:
        notice_type = row["고지형식"]
        household_type = row["세대유형"]
        pressure = row["사용(측정)압력"]
        rating = row["등급"]

        # 연소기정보가 문자열로 저장된 경우 JSON으로 파싱
        combustion_info = row["연소기정보"]
        if isinstance(combustion_info, str):
            try:
                combustion_info = json.loads(combustion_info.replace("'", '"'))
            except:
                pass

        # 연소기정보를 읽기 쉬운 형태로 변환
        if isinstance(combustion_info, list):
            combustion_text = ", ".join(
                [
                    f"{item.get('연소기명', 'N/A')}({item.get('수량', 'N/A')}대, {item.get('열량', 'N/A')}kcal)"
                    for item in combustion_info
                ]
            )
        else:
            combustion_text = str(combustion_info)

        query = f"""고지형식이 "{notice_type}" 일때, 아래 정보를 바탕으로 이상치를 판별해주세요.
    
세대유형: {household_type}
사용(측정)압력: {pressure}
등급: {rating}
연소기 정보: {combustion_text}"""

        return {
            "row_index": row.name,
            "query": query,
            "metadata": {
                "구분": row["구분"],
                "고지형식": notice_type,
                "세대유형": household_type,
                "사용(측정)압력": pressure,
                "등급": rating,
            },
        }
    except Exception as e:
        return {"row_index": row.name, "query": None, "error": str(e)}


def make_query(file_path, max_workers=4):
    """Excel 파일에서 모든 행에 대한 쿼리 리스트를 병렬처리로 생성"""
    # Excel 파일 읽기
    df = pd.read_excel(file_path)
    print(f"총 {len(df)}개의 행을 처리합니다.")

    query_results = []

    # 병렬처리로 쿼리 생성
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 행에 대해 쿼리 생성 작업 제출
        future_to_row = {
            executor.submit(create_single_query, row): idx for idx, row in df.iterrows()
        }

        # 완료된 작업들 수집
        for future in as_completed(future_to_row):
            try:
                result = future.result()
                query_results.append(result)
            except Exception as e:
                row_idx = future_to_row[future]
                query_results.append(
                    {
                        "row_index": row_idx,
                        "query": None,
                        "error": f"처리 중 오류 발생: {str(e)}",
                    }
                )

    # 행 인덱스 순서로 정렬
    query_results.sort(key=lambda x: x["row_index"])

    # 성공적으로 생성된 쿼리만 추출
    successful_queries = [
        result for result in query_results if result.get("query") is not None
    ]
    failed_queries = [result for result in query_results if result.get("query") is None]

    print(f"성공적으로 생성된 쿼리: {len(successful_queries)}개")
    if failed_queries:
        print(f"실패한 쿼리: {len(failed_queries)}개")
        for failed in failed_queries:
            print(
                f"  행 {failed['row_index']}: {failed.get('error', '알 수 없는 오류')}"
            )

    return successful_queries


# 테스트용 함수
def make_query_run():
    """테스트 실행 함수"""
    # file_path = "merged_data.xlsx"
    query_list = make_query(file_path="./merged_data.xlsx")

    # 결과 출력
    print(f"\n생성된 쿼리 예시:")
    for i, query_data in enumerate(query_list[:3]):  # 처음 3개만 출력
        print(f"\n--- 쿼리 {i+1} ---")
        print(f"구분: {query_data['metadata']['구분']}")
        print(f"쿼리:\n{query_data['query']}")
        print("-" * 50)

    return query_list


# if __name__ == "__main__":
#     test_make_query()
