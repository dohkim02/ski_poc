import asyncio

import queue

from _analyze import *
from _qna import *


# ✅ 진행률 추적 클래스
class ProgressTracker:
    def __init__(self):
        self.completed = 0
        self.total = 0
        self.progress_queue = queue.Queue()

    def update(self, completed: int, total: int):
        self.completed = completed
        self.total = total
        self.progress_queue.put((completed, total))

    def get_progress(self):
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None


# ✅ 실제 진행률을 추적하는 배치 처리 함수
async def batch_process_with_tracker(
    llm,
    all_data_chunks,
    output_file="final_results.jsonl",
    max_concurrency=20,
    progress_tracker=None,
):
    lock = asyncio.Lock()
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    semaphore = asyncio.Semaphore(max_concurrency)
    completed_tasks = 0
    total_tasks = len(all_data_chunks)

    if progress_tracker:
        progress_tracker.total = total_tasks

    async def run_with_semaphore(chunk):
        nonlocal completed_tasks
        async with semaphore:
            result = await async_llm(llm, chunk, lock, output_file)
            completed_tasks += 1

            # ✅ 진행률 업데이트
            if progress_tracker:
                progress_tracker.update(completed_tasks, total_tasks)

            return result

    tasks = [
        asyncio.create_task(run_with_semaphore(chunk)) for chunk in all_data_chunks
    ]

    results = await asyncio.gather(*tasks)
    return results


# ✅ 질의응답용 진행률 추적 함수
async def batch_llm_answers_with_tracker(llm, query, chunks, progress_tracker=None):
    results = []
    total = len(chunks)

    if progress_tracker:
        progress_tracker.total = total

    for i, chunk in enumerate(chunks):
        try:
            result = await async_llm_answer(
                llm, query, chunk
            )  # 이 함수는 _qna.py에 있다고 가정
            if result:
                results.append(result)

            # ✅ 진행률 업데이트
            if progress_tracker:
                progress_tracker.update(i + 1, total)

        except Exception as e:
            if progress_tracker:
                progress_tracker.update(i + 1, total)
            continue

    return results
