from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import json
from prompt import CLASSIFY_DATA_GROUP, JUDGE_BIZ_PROMPT
from utils import get_json, extract_group_data, get_data_from_txt
import os
import sys
import asyncio
from tqdm import tqdm

MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm


class Analyze:
    def __init__(
        self,
        llm,
        biz_gt_path="./_biz_group_gt.json",
        usage_gt_path="./_usage_gt.json",
    ):
        self.llm = llm
        self._biz_gt = get_json(biz_gt_path)
        self._usage_gt = get_json(usage_gt_path)

    async def classify_llm(self, data):
        class Classify(BaseModel):
            result: str = Field(
                description='분류된 그룹의 숫자. 반드시 문자열 숫자 하나여야 함. "0"부터 "11" 사이 중 하나.""'
            )

        structured_llm = self.llm.with_structured_output(Classify)
        input_data = str(data["업태"]) + "(" + str(data["업종"]) + ")"
        prompt = PromptTemplate.from_template(CLASSIFY_DATA_GROUP)
        chain = prompt | structured_llm
        result = await asyncio.to_thread(chain.invoke, {"data": input_data})

        return result.result

    async def judge_with_biz_llm(self, group_id, data):
        class Judge(BaseModel):
            result: str = Field(
                description="Ground Truth 데이터와 비교했을 때, 표준편차를 벗어나는지 확인하여 이상치인지 확인. 답변은 항상 '이상' or '정상'"
            )
            reason: str = Field(
                description="이상치라면, 이상치라고 판단한 이유. 반드시 해당 그룹의 데이터를 근거로 답변."
            )

        gt_data = extract_group_data(group_id, self._biz_gt)
        gt_to_extract = ["그룹", "사용량 패턴 평균", "사용량 패턴 표준편차"]
        gt_data = {k: gt_data[k] for k in gt_to_extract if k in gt_data}
        pattern_data = data["사용량_패턴"]
        gt_data_str = json.dumps(gt_data, ensure_ascii=False)

        structured_llm = self.llm.with_structured_output(Judge)
        prompt = PromptTemplate.from_template(JUDGE_BIZ_PROMPT)
        chain = prompt | structured_llm
        result = await asyncio.to_thread(
            chain.invoke, {"gt_data_str": gt_data_str, "data": pattern_data}
        )

        return result, gt_data

    async def process_single_item(self, data_item):
        group_id = await self.classify_llm(data_item)
        judge_result, gt_data = await self.judge_with_biz_llm(group_id, data_item)

        return {
            "judge_result": judge_result,
            "input_data": data_item,
            "gt_data": gt_data,
        }

    async def run_biz_judge(self, data_lst):
        """비동기로 모든 데이터를 처리하되 순서를 보장"""

        # 프로그레스바 초기화
        progress_bar = tqdm(total=len(data_lst), desc="Processing data", unit="item")

        results = []

        # 세마포어로 동시 실행 수 제한 (선택사항)
        semaphore = asyncio.Semaphore(50)  # 최대 50개 동시 실행

        async def process_with_progress(data_item):
            async with semaphore:
                result = await self.process_single_item(data_item)
                progress_bar.update(1)
                return result

        # 모든 작업을 비동기로 실행하되 순서 보장
        tasks = [process_with_progress(data_item) for data_item in data_lst]
        results = await asyncio.gather(*tasks)

        progress_bar.close()
        return results


# 사용 예시:
async def main():
    llm = initialize_llm("langchain_gpt4o")
    data_lst = get_data_from_txt("./preprocessed.txt")

    print(f"Total items to process: {len(data_lst)}")

    analyzer = Analyze(llm)
    results = await analyzer.run_biz_judge(data_lst)

    # 에러가 있는지 확인
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"Found {len(errors)} errors during processing")

    return results


# 실행
if __name__ == "__main__":
    results = asyncio.run(main())

    output_path = "./biz_results_llm_std_tuning.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(results):
            if isinstance(item, Exception):
                f.write(f"[{i}] Error: {str(item)}\n\n")
            else:
                judge_result = item["judge_result"]
                input_data = item["input_data"]
                gt_data = item["gt_data"]

                result_dict = judge_result.model_dump()  # ✅ Pydantic v2 대응

                if result_dict["result"] == "이상":
                    f.write(f"[{i}] 결과 (이상 감지):\n")
                    f.write(json.dumps(result_dict, ensure_ascii=False, indent=2))
                    f.write("\n\n[입력 데이터]:\n")
                    f.write(json.dumps(input_data, ensure_ascii=False, indent=2))
                    f.write("\n\n[기준 데이터]:\n")
                    f.write(json.dumps(gt_data, ensure_ascii=False, indent=2))
                    f.write("\n\n")
                else:
                    f.write(f"[{i}] 결과 (정상):\n")
                    f.write(json.dumps(result_dict, ensure_ascii=False, indent=2))
                    f.write("\n\n")
