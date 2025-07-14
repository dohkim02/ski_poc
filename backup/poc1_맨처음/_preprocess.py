import pandas as pd
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from _preprocess_exel import *


def chunk_list(data_list, chunk_size=10):
    return [data_list[i : i + chunk_size] for i in range(0, len(data_list), chunk_size)]


def write_chunk_line(file_path, chunk, lock):
    line = json.dumps(chunk, ensure_ascii=False)
    # 스레드가 동시에 파일에 쓰지 않도록 락을 사용!
    with lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def excel_to_txt_chunks_parallel(
    file_path, output_file="./test_chunks.txt", chunk_size=10
):
    import threading

    # 1️⃣ 엑셀 읽기
    df = pd.read_excel(file_path)

    # 2️⃣ JSON 변환
    data_list = json.loads(df.to_json(orient="records", force_ascii=False))

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


# if __name__ == "__main__":
#     # 1️⃣ 첫 번째 함수: 결측 제거 및 정렬 (sheet2 기준)
#     file_path = "../data/data1.xlsx"
#     output_path = drop_na(file_path)

#     # 2️⃣ 두 번째 함수: 다른 파일도 '구분' 일치시켜 새로 저장
#     new_output_path = data_split(file_path, output_path)

#     # 3️⃣ 세 번째 함수: 두 파일 병합
#     if os.path.exists(output_path) and os.path.exists(new_output_path):
#         print("파일 존재 확인: 두 파일 모두 존재하므로 병합 시작!")
#         merged_df = data_merge(output_path, new_output_path)

#         # 결과를 덮어씌우도록 저장
#         merged_output_path = "../data/preprocessed_data.xlsx"
#         merged_df.to_excel(merged_output_path, index=False)
#         print(f"병합된 데이터가 {merged_output_path}에 저장되었습니다.")
#     else:
#         print("⚠️ 병합할 파일이 존재하지 않습니다. 확인해주세요!")

#     get_txt = excel_to_txt_chunks_parallel(merged_output_path)
