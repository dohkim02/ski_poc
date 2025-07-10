import json
import os
import sys

MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)


def get_line_lengths(txt_path):
    lengths = []
    final = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            items = [x.strip() for x in line.strip().split(",") if x.strip()]
            lengths.append(len(items))
            final = final + items
    return final, lengths


def get_json_line_lengths(txt_path):
    lengths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    lengths.append(len(data))
                else:
                    print(f"⚠️ {i}번째 줄은 리스트가 아님: {type(data)}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 실패 (줄 {i}): {e}")
    return lengths


final, lengths = get_line_lengths("./clustering_result.txt")
# print("각 줄의 길이:", lengths)
# print("전체 줄 수:", len(lengths))
# print("전체 총 개수:", sum(lengths))


# lengths = get_json_line_lengths("./clustering_result.txt")
indexes = [i for i, val in enumerate(lengths) if val != 10]
print("각 줄 리스트 길이:", lengths)
print("전체 줄 수:", len(lengths))
print("전체 항목 수:", sum(lengths))
import pdb

pdb.set_trace()
