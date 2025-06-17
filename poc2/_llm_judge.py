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

    async def test_llm(self, data):
        data_num = len(data)
        template = """너는 월별 사용량 데이터를 분석하여, 검침 오류로 의심되는 이상 데이터를 찾아내는 AI이다.

        입력된 데이터는 총 {data_num}개의 월별 사용량 항목으로 구성되어 있다.
        이 중에서 너만의 판단 기준(예: 극단적인 수치, 급격한 변동, 계절적 불일치 등)을 기반으로,
        **검침 오류가 의심되는 데이터 상위 1%만 선택**해라.

        선정한 이상 데이터는 반드시 다음 항목을 포함해야 한다:
        - 식별자 또는 인덱스
        - 해당 데이터의 값 또는 구조
        - 이상치로 판단한 구체적인 이유 (수치적/패턴적 설명)

        [입력 데이터]
        {data}

        ⚠️ 주의:
        - 반드시 상위 1%만 골라야 하며, 사소한 편차는 제외
        - 설명은 간결하되 설득력 있게 써야 함
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm
        result = await asyncio.to_thread(
            chain.invoke, {"data_num": data_num, "data": data}
        )

        return result

    def chunk_by_ratio(self, data_lst, ratio):
        assert 0 < ratio <= 1, "ratio는 0보다 크고 1 이하의 값이어야 합니다."

        total = len(data_lst)
        chunk_size = max(1, int(total * ratio))

        return [data_lst[i : i + chunk_size] for i in range(0, total, chunk_size)]

    async def process_chunk(self, data_chunk):
        result = await self.test_llm(data_chunk)

        return result

    async def run_data_lst(self, data_lst):
        MAX_CHUNK_SIZE = 250
        total_len = len(data_lst)

        # 자동 ratio 조정
        if total_len <= MAX_CHUNK_SIZE:
            ratio = 1.0
        else:
            ratio = MAX_CHUNK_SIZE / total_len
            chunks = self.chunk_by_ratio(data_lst, ratio)

        # 전체 항목 수 기준으로 프로그레스 바 설정
        total_items = len(chunks)
        progress_bar = tqdm(total=total_items, desc="Processing data", unit="item")

        results = []

        # 세마포어로 청크 단위 병렬 수 제한
        semaphore = asyncio.Semaphore(10)  # 동시에 몇 개의 청크를 처리할지 (조절 가능)

        async def process_chunk_with_progress(chunk):
            async with semaphore:
                result = await self.process_chunk(chunk)
                progress_bar.update(len(chunk))  # 청크 크기만큼 진행 업데이트
                return result.content

        # 청크 단위 병렬 실행
        tasks = [process_chunk_with_progress(chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)

        progress_bar.close()

        # 결과 합치기 (flatten)
        for chunk_result in chunk_results:
            results.append(chunk_result)

        return results

    def reports_llm(self, results):
        template = """
        너는 이상 데이터 분석 결과들을 종합해 최종 보고서를 작성하는 전문가 AI이다.
        아래 "입력 리포트"는 서로 다른 LLM 분석 결과이다. 이들은 각각 검침 오류가 의심되는 이상 데이터를 식별한 리포트이며, 일부 중복 항목이나 유사한 판단 기준이 포함되어 있다.

        너의 임무는 아래 리포트들을 바탕으로 **중복 없이 깔끔하고 전문가스럽게 정리된 최종 보고서**를 작성하는 것이다.

        ### 작성 규칙:
        1. **중복된 항목**은 하나로 통합하되, 중요한 정보는 병합하여 유지한다.
        2. **데이터 항목**은 식별자(예: 구분), 값 요약, 이상 판단 이유를 포함해야 한다.
        3. 리포트 형식은 다음을 따라야 한다:
        - 제목 (예: "이상 데이터 검출 결과")
        - 항목별 정리 (번호, 구분, 값, 이상 사유)
        - 최종 결론
        4. **분석 어투는 전문적이고 간결하게 작성하라.**

        [입력 리포트]
        {results}
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm
        result = chain.invoke({"results": results})

        return result.content


# 사용 예시:
async def main():
    llm = initialize_llm("langchain_gpt4o")
    data_lst = get_data_from_txt("./preprocessed.txt")

    analyzer = Analyze(llm)
    results = await analyzer.run_data_lst(data_lst)
    print(len(results))
    # 에러가 있는지 확인
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"Found {len(errors)} errors during processing")

    final_ans = analyzer.reports_llm(results)

    return final_ans


# # 실행
# if __name__ == "__main__":
#     results = asyncio.run(main())
#     import pdb

#     pdb.set_trace()

# ans = analyzer.test_llm(data_lst)

# from pprint import pprint

# pprint(ans.content)
