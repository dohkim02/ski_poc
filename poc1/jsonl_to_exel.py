import pandas as pd
import json

# JSONL 파일 경로
jsonl_path = "./final_results.jsonl"

# 각 줄을 JSON으로 읽어 리스트에 저장
data = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]

# 판다스로 변환
df = pd.DataFrame(data)

# Excel로 저장
df.to_excel("./output.xlsx", index=False)
