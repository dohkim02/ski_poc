from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import json
from prompt import CLASSIFY_DATA_GROUP, JUDGE_BIZ__STD_PROMPT
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
                description="이상치라면, 이상치라고 판단한 구체적인 이유. 반드시 해당 그룹의 데이터를 근거로 수치적으로 답변."
            )

        gt_data = extract_group_data(group_id, self._biz_gt)
        gt_to_extract = ["그룹", "사용량 패턴 평균", "사용량 패턴 표준편차"]
        gt_data = {k: gt_data[k] for k in gt_to_extract if k in gt_data}
        pattern_data = data["사용량_패턴"]
        gt_data_str = json.dumps(gt_data, ensure_ascii=False)

        structured_llm = self.llm.with_structured_output(Judge)
        prompt = PromptTemplate.from_template(JUDGE_BIZ__STD_PROMPT)
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

    def reports_llm(self, results):
        template = """
        너는 이상 데이터 분석 결과들을 종합해 최종 보고서를 작성하는 전문가 AI이다.

        아래 "입력 리포트"는 서로 다른 {N}개의 LLM 분석 결과로, 각 리포트는 통계 기준(예: 평균, 표준편차)을 활용하여 **패턴에서 벗어난 이상 데이터를 식별**한 결과이다.  
        이 리포트들에는 일부 중복 항목이나 유사한 판단 기준이 포함되어 있으며, 모두 월별 사용 패턴을 포함하고 있다.

        너의 임무는 다음과 같다:
        - 입력된 리포트 중 **통계적으로 신뢰도 높고 대표적인 이상 사례들**을 바탕으로, **최종 보고서**를 작성하라.
        - 이상 사례의 판단 기준은 다음 중 하나 이상을 충족해야 한다:
            - 모든 월에서 평균보다 지속적으로 낮거나 높은 **전형적인 이상 패턴**
            - 특정 월에서 극단적 수치로 평균 ± 2.5σ 이상 벗어나는 **통계적 이상치**
            - 계절성 패턴을 완전히 무시하는 **비상식적 패턴**
            - 열량이나 사용량이 비정상적으로 0에 가깝거나 과도하게 높은 경우
            - 유사 업종 대비 이상 수치가 두드러지는 항목
        - 리포트 상에 드러나지 않도록 하되, 위 기준을 만족하는 **전체 중 약 1% 수준의 데이터만 사용하여 작성**하라.  
        (*단, 출력물에는 '1%' 또는 '선별'이라는 표현이 절대 포함되어선 안 된다.*)

        ### 작성 규칙:
        1. **중복된 항목**은 하나로 통합하되, 중요한 정보(이상 사유 등)는 병합하여 유지한다.
        2. 각 데이터 항목은 다음 정보를 반드시 포함해야 한다:
            - category
            - 구분(식별자)
            - 업태, 업종, 용도
            - 연소기 열량, 보일러 열량
            - 월별 사용량 / 평균 / 표준편차 (마크다운 표 형식) 
            - 이상 판단 사유 (구체적 이유)
        3. 리포트 형식은 다음을 따라야 한다:
            - 제목 (예: "이상 데이터 검출 결과")
            - 항목별 정리
            - 최종 결론 (전체 요약 및 시사점 포함)
        4. **전문적이고 간결한 어투**로 작성할 것.

        ---

        [입력 리포트]
        {results}
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm
        result = chain.invoke({"N": len(results), "results": results})

        return result.content


def save_results_to_txt(output_path, results):
    count = 0
    check_lst = []

    # 먼저 이상 결과 수집
    for item in results:
        if not isinstance(item, Exception):
            judge_result = item["judge_result"]
            result_dict = judge_result.model_dump()
            if result_dict["result"] == "이상":
                count += 1
                check_lst.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        # ✅ 상단에 이상 데이터 개수 기록
        f.write(f"⚠️ 이상 데이터 감지 개수: {count}건\n\n")

        for i, item in enumerate(check_lst):
            judge_result = item["judge_result"]
            input_data = item["input_data"]
            gt_data = item["gt_data"]

            result_dict = judge_result.model_dump()

            f.write(f"\n\n[결과 (이상 감지):]\n")
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=2))
            f.write(f"\n\n[확인 데이터 그룹: {gt_data['그룹']}]:\n")
            f.write(json.dumps(input_data, ensure_ascii=False, indent=2))

    return output_path


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

    outlier_data = [
        {"category": item["gt_data"]["그룹"], **item}
        for item in results
        if item["judge_result"].result == "이상"
    ]
    # import pdb

    # pdb.set_trace()
    # results[0]["gt_data"]["그룹"]
    output_path = "./llm_std_test.txt"
    output_path = save_results_to_txt(output_path, results)
    final_report = analyzer.reports_llm(outlier_data)
    return final_report


# 실행
if __name__ == "__main__":
    results = asyncio.run(main())
    import pdb

    pdb.set_trace()
