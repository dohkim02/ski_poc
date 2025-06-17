from prompt import QNA_SYSTEM_PROMPT
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


# LLM API 호출 부분 (동기 메서드)
def sync_call_llm(client, query, data_chunk):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": QNA_SYSTEM_PROMPT.format(
                    data=data_chunk,
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=4096,
        temperature=0,
    )
    return response.choices[0].message.content


async def async_llm_answer(llm, query, data_chunk, max_retries=5):
    client = llm

    for attempt in range(1, max_retries + 2):  # 첫 시도 + max_retries번 시도
        try:
            # ✅ 동기 메서드를 스레드로 실행!
            ai_response = await asyncio.to_thread(
                sync_call_llm, client, query, data_chunk
            )

            return ai_response  # 성공하면 리턴하고 끝!

        except Exception as e:
            print(f"🚨 오류 발생 (시도 {attempt}/{max_retries + 1}): {e}")
            if attempt == max_retries + 1:
                print("⚠️ 최대 재시도 횟수 초과! 이 chunk는 넘어감.")
                return None
            else:
                print("🔄 재시도 중...")
                await asyncio.sleep(2)  # 2초 쉬고 재시도 (필요하면 조절 가능)


async def batch_llm_answers(
    llm,
    query,
    all_data_chunks: List[List[Dict]],
    max_concurrency=20,
):

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(chunk):
        async with semaphore:
            return await async_llm_answer(llm, query, chunk)

    tasks = [
        asyncio.create_task(run_with_semaphore(chunk)) for chunk in all_data_chunks
    ]

    # ✅ gather로 병렬로 한방에 처리 (진짜 병렬 느낌!)
    results = await tqdm_asyncio.gather(*tasks, desc="진행상황")

    return results


def load_jsonl_chunks(file_path, chunk_size=20):
    chunks = []
    current_chunk = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line.strip())
            current_chunk.append(line_data)

            if len(current_chunk) == chunk_size:
                chunks.append(current_chunk)
                current_chunk = []

    # 마지막 남은 데이터 처리
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# if __name__ == "__main__":
#     llm = initialize_llm("gpt4o")

#     result_file = "./final_results.jsonl"
#     chunks = load_jsonl_chunks(result_file)

#     while True:
#         query = input("질문: ")

#         # ✅ 비동기 배치 처리
#         all_results = asyncio.run(batch_llm_answers(llm, query, chunks))

#         # ✅ None이 아닌 결과만 정리
#         all_results = [r for r in all_results if r is not None]

#         # ✅ 10개씩 나눠서 출력
#         chunk_size = 10
#         for i in range(0, len(all_results), chunk_size):
#             batch = all_results[i : i + chunk_size]
#             print("\n".join(batch))
#             input("\n--- 더 보려면 엔터를 누르세요 ---")
