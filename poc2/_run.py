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
import tempfile

# 모델 경로 설정 - Streamlit Cloud 호환
try:
    from model import initialize_llm
except ImportError:
    # 상위 디렉토리에서 찾기
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import initialize_llm


class Analyze:
    def __init__(
        self,
        llm,
        ground_truth_path=None,
    ):
        self.llm = llm
        self.ground_truth = None  # 항상 속성 생성

        # Streamlit Cloud 호환 방식으로 ground_truth 파일 찾기
        if ground_truth_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 가능한 파일 위치들 (Streamlit Cloud에서 더 안전한 순서)
            possible_locations = [
                os.path.join(current_dir, "group_index.json"),  # 같은 디렉토리
                os.path.join(
                    current_dir, "make_instruction", "group_index.json"
                ),  # 하위 디렉토리
                "group_index.json",  # 현재 작업 디렉토리
                "make_instruction/group_index.json",  # 상대 경로
                os.path.join(
                    os.getcwd(), "group_index.json"
                ),  # 현재 작업 디렉토리 절대경로
                os.path.join(os.getcwd(), "poc2", "group_index.json"),  # poc2 디렉토리
            ]

            print(f"🔍 Looking for group_index.json in the following locations:")
            for i, path in enumerate(possible_locations, 1):
                exists = os.path.exists(path)
                print(f"  {i}. {path} -> {'✓ FOUND' if exists else '✗ NOT FOUND'}")
                if exists:
                    ground_truth_path = path
                    print(f"✅ Using ground_truth file: {ground_truth_path}")
                    break

            if ground_truth_path is None:
                # 최후의 수단: 기본 구조로 빈 딕셔너리 초기화
                print(
                    "⚠️  WARNING: group_index.json not found. Creating empty ground truth."
                )
                self.ground_truth = self._create_default_ground_truth()
                return

        try:
            print(f"📄 Loading group_index.json from: {ground_truth_path}")
            self.ground_truth = get_json(ground_truth_path)
            print(
                f"✅ Ground truth loaded successfully with {len(self.ground_truth)} entries"
            )
        except Exception as e:
            print(f"❌ Error loading ground truth file: {str(e)}")
            print("🔄 Falling back to default ground truth")
            self.ground_truth = self._create_default_ground_truth()

    def _create_default_ground_truth(self):
        """기본 ground truth 구조 생성"""
        return {
            "A": {
                "건설업": {
                    "일반용1": {
                        "저압": {
                            "category": "건설업",
                            "standard": {
                                "1월": 100,
                                "2월": 100,
                                "3월": 100,
                                "4월": 100,
                                "5월": 100,
                                "6월": 100,
                                "7월": 100,
                                "8월": 100,
                                "9월": 100,
                                "10월": 100,
                                "11월": 100,
                                "12월": 100,
                            },
                            "data_num": 10,
                        }
                    }
                }
            }
        }

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
        grade = data.get("등급", "A")  # 기본값 A
        usage = data.get("용도", "일반용1")  # 기본값 일반용1
        # 안전한 키 접근으로 변경
        pressure = data.get("압력_그룹", "저압")  # 기본값을 "저압"으로 설정

        ground_truth = get_group_usage_info(
            self.ground_truth, grade, category, usage, pressure
        )
        import pdb

        pdb.set_trace()

        # ground_truth가 문자열인 경우 (에러 메시지) 기본 딕셔너리 반환
        if isinstance(ground_truth, str):
            print(
                f"Warning: get_group_usage_info returned error message: {ground_truth}"
            )
            # 기본 구조를 가진 딕셔너리 반환
            return {
                "category": f"({grade}, {category}, {usage}, {pressure})",
                "standard": {},
                "data_num": 0,
            }

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
            # 기존 컬럼 안전하게 삭제 (존재하는 컬럼만)
            columns_to_drop = [
                col for col in ["업태", "업종", "용도", "등급"] if col in df.columns
            ]
            if columns_to_drop:
                df = df.drop(columns_to_drop, axis=1)
            # DataFrame을 다시 딕셔너리로 변환
            data = df.iloc[0].to_dict()

            # ground_truth = json.dumps(ground_truth, ensure_ascii=False)
            gt = ground_truth["standard"]

            # 안전한 키 접근으로 변경
            years_data = data.get("3년치 데이터", {})
            input_data = get_latest_6month(years_data)
            # gt 딕셔너리에도 안전하게 접근
            standard_data = {month: gt.get(month, 0) for month in input_data.keys()}

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
                # 안전한 키 접근으로 변경
                years_data = outlier_item["input_data"].get("3년치 데이터", {})

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

    # Streamlit Cloud 호환: 임시 파일로 데이터 처리
    try:
        # 현재 디렉토리에서 preprocessed.txt 찾기
        current_dir = os.path.dirname(__file__)
        possible_data_paths = [
            os.path.join(current_dir, "preprocessed.txt"),
            "preprocessed.txt",
            os.path.join(os.getcwd(), "preprocessed.txt"),
            os.path.join(os.getcwd(), "poc2", "preprocessed.txt"),
        ]

        data_file_path = None
        for path in possible_data_paths:
            if os.path.exists(path):
                data_file_path = path
                break

        if data_file_path is None:
            print("❌ preprocessed.txt file not found in any expected location")
            return []

        print(f"📄 Loading data from: {data_file_path}")
        data_lst = get_data_from_txt(data_file_path)
        print(f"✅ Loaded {len(data_lst)} items")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return []

    print(f"🔄 Total items to process: {len(data_lst)}")

    analyzer = Analyze(llm)
    results = await analyzer.run_biz_judge(data_lst)

    # '이상'인 결과만 필터링 (judge_result가 비어있지 않은 경우)
    outlier_results = [
        item for item in results if item["judge_result"]  # 빈 리스트가 아닌 경우
    ]

    # outlier_results에 대해 pattern_checker 병렬 실행
    if outlier_results:
        print(f"🔍 Running pattern check on {len(outlier_results)} outlier cases...")
        outlier_results = await analyzer.run_pattern_check(outlier_results)

    # Streamlit Cloud 호환: 임시 파일로 결과 저장
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp_file:
            output_path = tmp_file.name
            write_outlier(output_path, outlier_results)
            print(f"💾 Outlier results saved to temp file: {output_path}")

        # pattern_checker 결과가 있는 경우 추가 분석 결과 저장
        write_post_process(outlier_results)
    except Exception as e:
        print(f"⚠️  Warning: Could not save results to file: {str(e)}")

    return results


# # 실행
if __name__ == "__main__":
    results = asyncio.run(main())
#     # import pdb

# pdb.set_trace()
