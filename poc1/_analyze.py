from prompt import ANALYZE_SYSTEM_PROMPT
import pandas as pd
import json
from pydantic import BaseModel
from typing import Dict, List
import sys
import os
import re
import asyncio
from tqdm.asyncio import tqdm_asyncio

# model.py가 있는 폴더 경로
MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm


class EvaluationInfo(BaseModel):
    result: str  # 판단 결과 (ex: "정상", "이상", 설명 포함 가능)
    reason: str  # 판단 이유


class EvaluationOutput(BaseModel):
    user_input: Dict  # 사용자 질문 전체 원본
    usage_evaluation: EvaluationInfo  # 용도 판단
    pressure_evaluation: EvaluationInfo  # 압력 판단
    final_judgement: str  # "정상", "이상", "주의"


# LLM API 호출 부분 (동기 메서드)
def sync_call(client, usage_burner, data_chunk):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ANALYZE_SYSTEM_PROMPT.format(
                    usage_burner=json.dumps(usage_burner, ensure_ascii=False, indent=2),
                    data=json.dumps(data_chunk, ensure_ascii=False, indent=2),
                ),
            },
        ],
        max_tokens=4096,
        temperature=0,
    )
    return response.choices[0].message.content


async def async_llm(llm, data_chunk, lock, output_file, max_retries=5):
    client = llm

    with open("./usage_burner.json", "r", encoding="utf-8") as f:
        usage_burner = json.load(f)

    for attempt in range(1, max_retries + 2):  # 첫 시도 + max_retries번 시도
        try:
            # ✅ 동기 메서드를 스레드로 실행!
            ai_response = await asyncio.to_thread(
                sync_call, client, usage_burner, data_chunk
            )

            cleaned_response = re.sub(r"```json|```", "", ai_response).strip()
            parsed_json_list = json.loads(cleaned_response)

            structured_data: List[EvaluationOutput] = [
                EvaluationOutput(**entry) for entry in parsed_json_list
            ]

            async with lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    for entry in structured_data:
                        line = json.dumps(entry.model_dump(), ensure_ascii=False)
                        f.write(line + "\n")

            return structured_data  # 성공하면 리턴하고 끝!

        except Exception as e:
            print(f"🚨 오류 발생 (시도 {attempt}/{max_retries + 1}): {e}")
            if attempt == max_retries + 1:
                print("⚠️ 최대 재시도 횟수 초과! 이 chunk는 넘어감.")
                return None
            else:
                print("🔄 재시도 중...")
                await asyncio.sleep(2)  # 2초 쉬고 재시도 (필요하면 조절 가능)


def get_data_from_txt(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line.strip())
            data_list.append(line_data)
    return data_list


async def batch_process(
    llm,
    all_data_chunks: List[List[Dict]],
    output_file="final_results.jsonl",
    max_concurrency=20,
):
    lock = asyncio.Lock()
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(chunk):
        async with semaphore:
            return await async_llm(llm, chunk, lock, output_file)

    tasks = [
        asyncio.create_task(run_with_semaphore(chunk)) for chunk in all_data_chunks
    ]

    # ✅ gather로 병렬로 한방에 처리 (진짜 병렬 느낌!)
    results = await tqdm_asyncio.gather(*tasks, desc="진행상황 (한방에!)")

    return results


def count_jsonl_lines_simple(file_path):
    """
    가장 간단한 방법: 라인 수 세기
    각 라인이 하나의 JSON 객체라고 가정
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())  # 빈 라인 제외
        return count
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return 0
    except Exception as e:
        print(f"오류 발생: {e}")
        return 0


# if __name__ == "__main__":
#     llm = initialize_llm("gpt4o")
#     result_file = "./final_results.jsonl"
#     # ✅ data_chunks.txt에서 청크 불러오기
#     chunk_file = "./test_chunk.txt"
#     all_data_chunks = get_data_from_txt(chunk_file)

#     # ✅ 비동기 배치 처리 실행 (Deprecation 경고 없이!)
#     asyncio.run(batch_process(llm, all_data_chunks, output_file=result_file))

#     print("\n✅ 모든 결과가 'final_results.jsonl'에 저장되었습니다!")

#     count_line = count_jsonl_lines_simple(result_file)
#     print(f"이상치라고 판단한 데이터의 총 개수: {count_line}")
