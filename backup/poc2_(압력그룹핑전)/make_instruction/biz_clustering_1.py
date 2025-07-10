import asyncio
import pandas as pd
import os
import sys
from typing import List
from pydantic import BaseModel, Field
from prompt import CLUSTERING_PROMPT
from utils import get_data_from_txt, get_biz_lst, get_exel_with_biz_lst
from langchain_core.prompts import PromptTemplate
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

MODEL_PATH = os.path.abspath("../../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm


class Clustering:
    def __init__(self, llm):
        self.llm = llm


class Clustering:
    def __init__(self, llm):
        self.llm = llm

    async def clustering_llm(self, biz_lst, pbar=None):
        class Classify(BaseModel):
            result: List[str] = Field(description="카테고리화 된 결과값들의 리스트")

        output_num = len(biz_lst)
        structured_llm = self.llm.with_structured_output(Classify)
        prompt = PromptTemplate.from_template(CLUSTERING_PROMPT)
        chain = prompt | structured_llm

        result = await asyncio.to_thread(
            chain.invoke, {"output_num": output_num, "biz_lst": biz_lst}
        )

        if pbar:
            pbar.update(1)

        return result.result

    async def clustering_run_async(
        self, whole_biz_lst: List[List[str]]
    ) -> List[List[str]]:
        results = []
        # tqdm 객체 생성
        with tqdm(total=len(whole_biz_lst), desc="Clustering 진행 중") as pbar:
            # gather를 순서대로 실행하되, pbar를 넘겨서 내부에서 update
            tasks = [self.clustering_llm(lst, pbar=pbar) for lst in whole_biz_lst]
            results = await asyncio.gather(*tasks)

        return results


# ⬇️ 비동기 실행 진입점
async def main():
    biz_lst = get_data_from_txt("./category_list.txt")  # 리스트 리스트 형태여야 함
    llm = initialize_llm("langchain_gpt4o")
    cls = Clustering(llm)

    # 실제 비동기 실행
    ans = await cls.clustering_run_async(biz_lst)

    # ✅ 결과를 한 줄씩 저장
    output_path = "./clustering_result.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in ans:
            # 리스트 항목들을 탭 또는 쉼표로 구분해 한 줄로 저장
            f.write(", ".join(line) + "\n")

    print(f"\n✅ 결과가 저장되었습니다: {output_path}")


# # ⬇️ 진입점 실행
if __name__ == "__main__":
    # asyncio.run(main())

    # # 사용 예시
    # get_exel_with_biz_lst(
    #     txt_path="./clustering_result.txt",
    #     xlsx_path="./preprocessed.xlsx",
    #     output_path="./group_biz_with_12.xlsx",
    )

    import pdb

    pdb.set_trace()
    pass
