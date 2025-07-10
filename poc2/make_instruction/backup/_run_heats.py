from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import json
from prompt import CLASSIFY_DATA_GROUP, JUDGE_WITH_GROUND_TRUTH_PROMPT, REPORTER_PROMPT
from utils import (
    get_json,
    extract_group_data,
    get_data_from_txt,
    get_group_usage_info,
    get_heat_input_gt,
)
import os
import sys
import asyncio
from tqdm import tqdm
import pandas as pd

MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm


class Analyze:
    def __init__(
        self,
        llm,
        ground_truth_path=os.path.join(
            os.path.dirname(__file__),
            "./make_instruction/group_biz_with_usage_heat.json",
        ),
    ):
        self.llm = llm
        self.ground_truth = get_json(ground_truth_path)

    # 업태와 업종을 기반으로 그룹 분류 후, 용도 파악하여 기준데이터 불러오기
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
        category_lst = {
            "0": "건설업",
            "1": "교육업",
            "2": "금융부동산업",
            "3": "기타",
            "4": "도소매업",
            "5": "비영리법인",
            "6": "서비스업",
            "7": "숙박 및 음식점업",
            "8": "시설관리업",
            "9": "운수업",
            "10": "정보통신업",
            "11": "제조업",
        }
        group_id = result.result

        category = category_lst[group_id]

        heat_category = get_heat_input_gt(data["열량"])

        ground_truth = get_group_usage_info(
            self.ground_truth, heat_category, category, data["용도"]
        )

        return ground_truth

    async def judge_with_biz_llm(self, ground_truth, data):
        class Judge(BaseModel):
            result: str = Field(
                description=(
                    "기준 데이터의 월별 사용량 median과 IQR 값을 기준으로, "
                    "각 월별 사용량이 이상 범위 (median ± 1.5 * IQR)를 벗어나는지 판단합니다. "
                    "이상 범위를 벗어난 월이 '월 순서상 연속하여 3개월 이상' 발생한 경우 '이상'을 반환하고, "
                    "그 외에는 '정상'을 반환합니다. "
                    "연속되지 않은 이상치는 개수와 관계없이 '정상'으로 간주합니다. "
                    "결과 값은 반드시 '이상' 또는 '정상' 중 하나여야 합니다."
                )
            )
            reason: str = Field(
                description=(
                    "'result'가 '이상'인 경우에만 채워집니다. "
                    "이상치로 판단된 월, 해당 월의 실제 값, 두 기준 각각의 허용 범위 (median ± 1.5 * IQR), "
                    "그리고 얼마나 벗어났는지를 수치적으로 서술합니다. "
                    "하나 이상의 기준과 월별 값이 어떻게 기준을 벗어났는지를 명확하게 서술해야 합니다. "
                    "'정상'인 경우 이 필드는 빈 문자열로 반환합니다."
                )
            )

        # 딕셔너리를 DataFrame으로 변환 (단일 행)
        df = pd.DataFrame([data])

        # 'heat_input' 컬럼 생성 (보일러 열량 + 연소기 열량)
        df["category"] = ground_truth["category"]
        # 기존 컬럼 삭제
        df = df.drop(
            [
                "업태",
                "업종",
                "용도",
                "보일러 열량",
                "보일러 대수",
                "연소기 열량",
                "연소기 대수",
            ],
            axis=1,
        )
        # DataFrame을 다시 딕셔너리로 변환
        data = df.iloc[0].to_dict()

        # ground_truth = json.dumps(ground_truth, ensure_ascii=False)
        ground_truth_median = json.dumps(ground_truth["median"], ensure_ascii=False)
        structured_llm = self.llm.with_structured_output(Judge)
        prompt = PromptTemplate.from_template(JUDGE_WITH_GROUND_TRUTH_PROMPT)
        chain = prompt | structured_llm
        result = await asyncio.to_thread(
            chain.invoke,
            {
                "ground_truth": ground_truth_median,
                "data": data["사용량_패턴"],
            },
        )

        return result

    async def process_single_item(self, data_item):
        ground_truth = await self.classify_llm(data_item)
        judge_result = await self.judge_with_biz_llm(ground_truth, data_item)

        return {
            "judge_result": judge_result,
            "input_data": data_item,
            "ground_truth": ground_truth,
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
    data_lst = get_data_from_txt(
        os.path.join(os.path.dirname(__file__), "preprocessed.txt")
    )

    print(f"Total items to process: {len(data_lst)}")

    analyzer = Analyze(llm)
    results = await analyzer.run_biz_judge(data_lst)

    # '이상'인 결과만 필터링
    outlier_results = [
        item for item in results if item["judge_result"].result == "이상"
    ]

    print(
        f"Found {len(outlier_results)} outlier cases out of {len(results)} total cases"
    )

    # txt 파일로 저장
    output_path = os.path.join(os.path.dirname(__file__), "outlier_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"이상 데이터 분석 결과 ({len(outlier_results)}건)\n")
        f.write("=" * 50 + "\n\n")

        for i, item in enumerate(outlier_results, 1):
            f.write(f"[{i}번째 이상 사례]\n")
            f.write(f"결과: {item['judge_result'].result}\n")
            f.write(f"이유: {item['judge_result'].reason}\n")
            f.write(f"기준 데이터: {item['ground_truth']}\n")
            f.write(f"입력 데이터: {item['input_data']}\n")
            f.write("-" * 30 + "\n\n")

    print(f"Outlier results saved to: {output_path}")
    return results


# 실행
if __name__ == "__main__":
    results = asyncio.run(main())
    import pdb

    pdb.set_trace()
