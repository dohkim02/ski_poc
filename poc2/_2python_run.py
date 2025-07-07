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
import ast

MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm


class Analyze:
    def __init__(
        self,
        llm,
        ground_truth_path=os.path.join(
            os.path.dirname(__file__),
            "./make_instruction/group_biz_with_usage_heat_optimize.json",
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

    def find_outlier(
        self,
        standard_data: dict,
        new_data: dict,
        min_consecutive: int = 3,
        percentage=0.5,
    ) -> list:
        """
        기준 데이터 대비 신규 데이터가 50% 미만인 월을 찾아,
        연속으로 min_consecutive개월 이상인 구간만 반환하는 함수.
        단, new_data 값이 0.0인 월은 제외.

        Args:
            standard_data (dict): 기준 데이터 (예: {"1월": 530.5, ...})
            new_data (dict): 신규 데이터 (예: {"1월": 287.96, ...})
            min_consecutive (int): 최소 연속 개월 수 (기본값: 3)

        Returns:
            List of lists: 연속 구간별 월 리스트 (예: [['6월', '7월', '8월', '9월']])
        """
        months_order = [
            "1월",
            "2월",
            "3월",
            "4월",
            "5월",
            "6월",
            "7월",
            "8월",
            "9월",
            "10월",
            "11월",
            "12월",
        ]

        under_80_flags = [
            new_data[month] < standard_data[month] * percentage
            and new_data[month] != 0.0
            for month in months_order
        ]

        result_sequences = []
        start = None

        for i, flag in enumerate(under_80_flags):
            if flag:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_consecutive:
                    result_sequences.append(months_order[start:i])
                start = None

        # 마지막까지 연속된 경우
        if start is not None and len(months_order) - start >= min_consecutive:
            result_sequences.append(months_order[start:])

        return result_sequences

    async def judge_with_biz_llm(self, ground_truth, data):
        try:
            # ground_truth가 문자열인 경우 JSON으로 파싱 시도
            if isinstance(ground_truth, str):
                try:
                    ground_truth = json.loads(ground_truth)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse ground_truth as JSON: {ground_truth}"
                    )
                    return []

            # ground_truth가 딕셔너리가 아닌 경우 처리
            if not isinstance(ground_truth, dict):
                print(
                    f"Warning: ground_truth is not a dictionary: {type(ground_truth)}"
                )
                return []

            # 필요한 키들이 있는지 확인
            required_keys = ["category", "median", "data_num"]
            missing_keys = [key for key in required_keys if key not in ground_truth]
            if missing_keys:
                print(f"Warning: Missing keys in ground_truth: {missing_keys}")
                return []

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
            standard_data = ground_truth["median"]
            input_data = ast.literal_eval(data["사용량_패턴"])
            if ground_truth["data_num"] > 1000:
                result = self.find_outlier(standard_data, input_data)
            else:
                result = []
            return result

        except Exception as e:
            print(f"Error in judge_with_biz_llm: {str(e)}")
            print(f"ground_truth type: {type(ground_truth)}")
            print(f"ground_truth value: {ground_truth}")
            return []

    async def process_single_item(self, data_item):
        ground_truth = await self.classify_llm(data_item)
        judge_result = await self.judge_with_biz_llm(ground_truth, data_item)
        keys_to_remove = ["연소기 대수", "보일러 열량", "보일러 대수", "연소기 열량"]
        input_data = {k: v for k, v in data_item.items() if k not in keys_to_remove}

        return {
            "judge_result": judge_result,
            "input_data": input_data,
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
    # import pdb

    # pdb.set_trace()
    results = await analyzer.run_biz_judge(data_lst)

    # '이상'인 결과만 필터링 (judge_result가 비어있지 않은 경우)
    outlier_results = [
        item for item in results if item["judge_result"]  # 빈 리스트가 아닌 경우
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

            # judge_result가 리스트 형태이므로 적절히 처리
            consecutive_months = item["judge_result"]
            if consecutive_months:
                f.write(f"결과: 이상\n")
                f.write(f"연속 이상 구간: {consecutive_months}\n")

                # 각 연속 구간별로 상세 정보 출력
                for j, month_sequence in enumerate(consecutive_months, 1):
                    f.write(
                        f"  구간 {j}: {' → '.join(month_sequence)} ({len(month_sequence)}개월 연속)\n"
                    )
            else:
                f.write(f"결과: 정상\n")

            # ground_truth의 data_num 정보 추가
            f.write(f"기준 데이터 샘플 수: {item['ground_truth']['data_num']}건\n")
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
