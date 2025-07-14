import pandas as pd
import json


def get_usage_to_burnerlist(file_path):
    # 엑셀파일 읽기
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # 1️⃣ '연소기명'을 인덱스로 세팅
    df = df.set_index("연소기명")

    # 2️⃣ 각 용도별로 'O'인 연소기명만 리스트화
    usage_dict = {}
    for usage in df.columns:
        burners = df.index[df[usage] == "O"].tolist()
        if burners:  # 비어있는 리스트는 생략
            usage_dict[usage] = burners

    return usage_dict


if __name__ == "__main__":
    file_path = "/Users/a10780/Desktop/poc/data/용도+압력 판단가이드.xlsx"  # 너의 파일 경로로 바꿔줘!
    usage_burner_dict = get_usage_to_burnerlist(file_path)

    # # 3️⃣ 예쁘게 출력
    # print("\n=== 용도별 연소기명 리스트 ===")
    # for usage, burners in usage_burner_dict.items():
    #     print(f"{usage}: {burners}")

    # 4️⃣ 필요하면 JSON으로 저장
    with open("usage_burner.json", "w", encoding="utf-8") as f:
        json.dump(usage_burner_dict, f, ensure_ascii=False, indent=2)
