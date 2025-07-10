# File version: 2.1 - Fixed group_index.json path resolution for Streamlit Cloud
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
from langchain_core.prompts import PromptTemplate
import json
from prompt import (
    CLASSIFY_DATA_GROUP,
    PATTERN_CHECK_PROMPT,
)
from utils import (
    get_json,
    get_latest_6month,
    get_previous_monthes,
    get_data_from_txt,
    get_group_usage_info,
    write_outlier,
    write_post_process,
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
        ground_truth_path=None,
    ):
        self.llm = llm

        # 더 간단하고 직접적인 방법으로 경로 찾기
        if ground_truth_path is None:
            # Streamlit Cloud 환경을 고려한 경로
            import streamlit as st

            # 현재 파일 기준으로 상대 경로 구성
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ground_truth_path = os.path.join(
                current_dir, "make_instruction", "group_index.json"
            )

            print(f"=== DEBUGGING FILE SYSTEM ===")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Current file (__file__): {__file__}")
            print(f"Current file directory: {current_dir}")
            print(f"Primary path: {ground_truth_path}")

            # 파일 시스템 구조 확인
            def explore_directory(path, max_depth=3, current_depth=0):
                items = []
                if current_depth >= max_depth:
                    return items
                try:
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            items.append(f"{'  ' * current_depth}📁 {item}/")
                            items.extend(
                                explore_directory(
                                    item_path, max_depth, current_depth + 1
                                )
                            )
                        else:
                            items.append(f"{'  ' * current_depth}📄 {item}")
                except PermissionError:
                    items.append(f"{'  ' * current_depth}❌ Permission denied")
                except Exception as e:
                    items.append(f"{'  ' * current_depth}❌ Error: {e}")
                return items

            print("\n=== EXPLORING FILE SYSTEM ===")
            print("Root directory structure:")
            root_items = explore_directory("/mount/src/ski_poc", max_depth=2)
            for item in root_items[:20]:  # Limit output
                print(item)

            print(f"\nCurrent directory contents:")
            current_items = explore_directory(current_dir, max_depth=2)
            for item in current_items[:15]:
                print(item)

            # group_index.json 파일을 재귀적으로 찾기
            print(f"\n=== SEARCHING FOR group_index.json ===")

            def find_file(directory, filename):
                found_paths = []
                try:
                    for root, dirs, files in os.walk(directory):
                        if filename in files:
                            found_paths.append(os.path.join(root, filename))
                        # 너무 깊이 들어가지 않도록 제한
                        if len(found_paths) > 5:
                            break
                except Exception as e:
                    print(f"Error walking directory {directory}: {e}")
                return found_paths

            # 여러 루트에서 파일 찾기
            search_roots = ["/mount/src/ski_poc", current_dir, os.getcwd()]
            all_found_paths = []

            for search_root in search_roots:
                if os.path.exists(search_root):
                    found = find_file(search_root, "group_index.json")
                    all_found_paths.extend(found)
                    print(f"Searching in {search_root}: {found}")

            # 파일이 존재하지 않으면 다른 경로들 시도
            if not os.path.exists(ground_truth_path):
                print(f"Primary path not found, trying alternatives...")

                alternative_paths = [
                    "/mount/src/ski_poc/poc2/make_instruction/group_index.json",
                    "/mount/src/ski_poc/make_instruction/group_index.json",
                    os.path.join(
                        current_dir,
                        "..",
                        "poc2",
                        "make_instruction",
                        "group_index.json",
                    ),
                    os.path.join(os.getcwd(), "make_instruction", "group_index.json"),
                    "make_instruction/group_index.json",
                ]

                # 찾은 파일들도 시도
                alternative_paths.extend(all_found_paths)

                for alt_path in alternative_paths:
                    print(
                        f"Trying: {alt_path} -> {'EXISTS' if os.path.exists(alt_path) else 'NOT FOUND'}"
                    )
                    if os.path.exists(alt_path):
                        ground_truth_path = alt_path
                        print(f"SUCCESS: Found at {alt_path}")
                        break
                else:
                    print(f"\n=== FINAL DEBUG INFO ===")
                    print(f"Could not find group_index.json anywhere!")
                    print(f"Searched paths: {alternative_paths}")
                    print(f"Found files named group_index.json: {all_found_paths}")

                    raise FileNotFoundError(
                        f"group_index.json not found. Tried paths: {alternative_paths}"
                    )

        print(f"Loading group_index.json from: {ground_truth_path}")
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
        grade = data["등급"]
        usage = data["용도"]
        pressure = data["압력_그룹"]

        ground_truth = get_group_usage_info(
            self.ground_truth, grade, category, usage, pressure
        )

        return ground_truth

    def find_outlier(
        self,
        standard_data: dict,
        new_data: dict,
        min_consecutive: int = 3,
        percentage=0.7,
    ) -> list:
        """
        기준 데이터 대비 신규 데이터가 하한 미만이거나 상한 초과인 월을 찾아,
        연속으로 min_consecutive개월 이상인 구간만 반환하는 함수.
        단, new_data 값이 0.0인 월은 제외.

        0.0이 3개월 이상 연속으로 나타나는 경우 빈 리스트를 반환.

        Args:
            standard_data (dict): 기준 데이터 (예: {"1월": 530.5, ...})
            new_data (dict): 신규 데이터 (예: {"1월": 287.96, ...})
            min_consecutive (int): 최소 연속 개월 수 (기본값: 3)
            percentage (float): 하한 비율 (기본값: 0.7, 상한은 1/percentage)

        Returns:
            List of dicts: 연속 구간별 정보
            (예: [{'months': ['6월', '7월', '8월'], 'types': ['하한', '하한', '상한']}])
        """
        # 월 이름을 숫자로 매핑
        month_to_num = {
            "1월": 1,
            "2월": 2,
            "3월": 3,
            "4월": 4,
            "5월": 5,
            "6월": 6,
            "7월": 7,
            "8월": 8,
            "9월": 9,
            "10월": 10,
            "11월": 11,
            "12월": 12,
        }

        # 실제 데이터에 있는 월들만 추출하고 순서대로 정렬
        available_months = list(set(standard_data.keys()) & set(new_data.keys()))
        if not available_months:
            return []

        # 월을 숫자 순서대로 정렬
        months_order = sorted(available_months, key=lambda x: month_to_num[x])

        # 데이터 개수가 min_consecutive보다 작으면 빈 리스트 반환
        if len(months_order) < min_consecutive:
            return []

        # 0.0이 연속으로 나타나는지 체크
        consecutive_zeros = 0
        max_consecutive_zeros = 0

        for month in months_order:
            if new_data[month] == 0.0:
                consecutive_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
            else:
                consecutive_zeros = 0

        # 0.0이 min_consecutive-1개월 이상 연속으로 나타나면 빈 리스트 반환
        if max_consecutive_zeros >= min_consecutive - 1:
            return []

        # 상한 비율 계산 (하한의 역수)
        upper_percentage = 1.5

        # 각 월의 이상치 여부와 타입(하한/상한) 체크
        outlier_info = []
        for month in months_order:
            if new_data[month] == 0.0:
                outlier_info.append({"is_outlier": False, "type": None})
            elif new_data[month] < standard_data[month] * percentage:
                outlier_info.append({"is_outlier": True, "type": "하한"})
            elif new_data[month] > standard_data[month] * upper_percentage:
                outlier_info.append({"is_outlier": False, "type": "상한"})
            else:
                outlier_info.append({"is_outlier": False, "type": None})

        # 마지막 min_consecutive개월이 이상치가 아닌지 확인 (데이터 길이가 충분한 경우만)
        last_months_not_outliers = True
        if len(outlier_info) >= min_consecutive:
            last_months_not_outliers = all(
                not outlier_info[i]["is_outlier"] for i in range(-min_consecutive, 0)
            )

        # 전체 월이 모두 이상치인지 확인
        all_months_are_outliers = all(info["is_outlier"] for info in outlier_info)

        result_sequences = []
        start = None

        for i, info in enumerate(outlier_info):
            if info["is_outlier"]:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_consecutive:
                    months_in_sequence = months_order[start:i]
                    types_in_sequence = [
                        outlier_info[j]["type"] for j in range(start, i)
                    ]
                    result_sequences.append(
                        {"months": months_in_sequence, "types": types_in_sequence}
                    )
                start = None

        # 마지막까지 연속된 경우 처리
        if (
            start is not None
            and len(months_order) - start >= min_consecutive
            and not (last_months_not_outliers and len(outlier_info) >= min_consecutive)
            and not all_months_are_outliers
        ):
            months_in_sequence = months_order[start:]
            types_in_sequence = [
                outlier_info[j]["type"] for j in range(start, len(months_order))
            ]
            result_sequences.append(
                {"months": months_in_sequence, "types": types_in_sequence}
            )

        # 전체가 이상치인 경우 빈 리스트 반환
        if all_months_are_outliers:
            return []

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
                    return [], {}, {}

            # ground_truth가 딕셔너리가 아닌 경우 처리
            if not isinstance(ground_truth, dict):
                print(
                    f"Warning: ground_truth is not a dictionary: {type(ground_truth)}"
                )
                return [], {}, {}

            # 필요한 키들이 있는지 확인
            required_keys = ["category", "standard", "data_num"]
            missing_keys = [key for key in required_keys if key not in ground_truth]
            if missing_keys:
                print(f"Warning: Missing keys in ground_truth: {missing_keys}")
                return [], {}, {}

            # 딕셔너리를 DataFrame으로 변환 (단일 행)
            df = pd.DataFrame([data])

            # 'heat_input' 컬럼 생성 (보일러 열량 + 연소기 열량)
            df["category"] = ground_truth["category"]
            # 기존 컬럼 삭제
            df = df.drop(
                ["업태", "업종", "용도", "등급"],
                axis=1,
            )
            # DataFrame을 다시 딕셔너리로 변환
            data = df.iloc[0].to_dict()

            # ground_truth = json.dumps(ground_truth, ensure_ascii=False)
            gt = ground_truth["standard"]

            input_data = get_latest_6month(data["3년치 데이터"])
            standard_data = {month: gt[month] for month in input_data.keys()}

            if ground_truth["data_num"] > 100:
                result = self.find_outlier(standard_data, input_data)
            else:
                result = []

            # 실제 비교에 사용된 데이터들도 함께 반환
            return result, standard_data, input_data

        except Exception as e:
            print(f"Error in judge_with_biz_llm: {str(e)}")
            print(f"ground_truth type: {type(ground_truth)}")
            print(f"ground_truth value: {ground_truth}")
            return [], {}, {}

    async def process_single_item(self, data_item):
        ground_truth = await self.classify_llm(data_item)
        judge_result, standard_data, input_data = await self.judge_with_biz_llm(
            ground_truth, data_item
        )

        return {
            "judge_result": judge_result,
            "input_data": data_item,
            "ground_truth": ground_truth,
            "standard_data": standard_data,  # 실제 비교에 사용된 기준 데이터
            "comparison_input_data": input_data,  # 실제 비교에 사용된 입력 데이터
        }

    async def pattern_checker(self, years_data):
        class Pattern(BaseModel):
            result: Literal["yes", "no"] = Field(
                description='이전년도들의 데이터 패턴을 분석하여 최근 데이터 패턴의 이상치 유무 확인. "yes" 또는 "no"로 답변'
            )
            reason: Optional[str] = Field(
                default=None,
                description='"result"가 "yes"인 경우에만 이상하다고 판단한 이유를 자세히 설명. 반드시 한국어로 답변해.',
            )

        keys = list(years_data.keys())
        values = list(years_data.values())

        # 안전한 데이터 변환 함수
        def safe_eval(value):
            if isinstance(value, (dict, list)):
                return value
            elif isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
            else:
                return value

        structured_llm = self.llm.with_structured_output(Pattern)
        prompt = PromptTemplate.from_template(PATTERN_CHECK_PROMPT)
        chain = prompt | structured_llm
        result = await asyncio.to_thread(
            chain.invoke,
            {
                "key0": keys[0],
                "key1": keys[1],
                "key2": keys[2],
                "key3": keys[3],
                "value0": safe_eval(values[0]),
                "value1": safe_eval(values[1]),
                "value2": safe_eval(values[2]),
                "value3": safe_eval(values[3]),
            },
        )

        return result

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

    async def run_pattern_check(self, outlier_results):
        """outlier_results에 대해 pattern_checker를 병렬로 실행"""

        # 프로그레스바 초기화
        progress_bar = tqdm(
            total=len(outlier_results), desc="Pattern checking", unit="item"
        )

        results = []

        # 세마포어로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(50)  # 최대 50개 동시 실행

        async def process_pattern_check(outlier_item):
            async with semaphore:
                # standard_6_month = outlier_item["standard_data"]

                # latest_6_month_data = outlier_item["comparison_input_data"]
                standard = outlier_item["ground_truth"]["standard"]
                years_data = outlier_item["input_data"]["3년치 데이터"]

                # 안전한 데이터 변환
                if isinstance(years_data, str):
                    try:
                        years_data = ast.literal_eval(years_data)
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse years_data: {years_data}")
                        progress_bar.update(1)
                        return outlier_item
                elif not isinstance(years_data, dict):
                    print(
                        f"Warning: years_data is not a valid format: {type(years_data)}"
                    )
                    progress_bar.update(1)
                    return outlier_item

                rest_month_data = get_previous_monthes(years_data)
                pattern_result = await self.pattern_checker(years_data)
                progress_bar.update(1)

                # 기존 outlier_item에 pattern_result 추가
                result_item = outlier_item.copy()
                result_item["pattern_result"] = pattern_result
                return result_item

        # 모든 작업을 비동기로 실행하되 순서 보장
        tasks = [
            process_pattern_check(outlier_item) for outlier_item in outlier_results
        ]
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

    # with open("./post_test.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    # outlier_results = ast.literal_eval(text)

    # outlier_results에 대해 pattern_checker 병렬 실행
    if outlier_results:
        print(f"Running pattern check on {len(outlier_results)} outlier cases...")
        outlier_results = await analyzer.run_pattern_check(outlier_results)

    # print(
    #     f"Found {len(outlier_results)} outlier cases out of {len(results)} total cases"
    # )

    # # txt 파일로 저장
    output_path = os.path.join(os.path.dirname(__file__), "outlier_results.txt")
    write_outlier(output_path, outlier_results)
    print(f"Outlier results saved to: {output_path}")
    # pattern_checker 결과가 있는 경우 추가 분석 결과 저장
    write_post_process(outlier_results)
    # return results
    return results


# # 실행
if __name__ == "__main__":
    results = asyncio.run(main())
#     # import pdb

# pdb.set_trace()
