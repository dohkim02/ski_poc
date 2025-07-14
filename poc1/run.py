import sys
import os
import numpy as np
from typing import List
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd

# 모델 경로 추가
MODEL_PATH = os.path.abspath("../")
sys.path.append(MODEL_PATH)
from model import initialize_embedding, initialize_llm
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import asyncio
from typing import Literal, Optional
from prompt import JUDGE_PROMPT

# embedding.py에서 수정된 함수 사용
from embedding import load_faiss_data
from make_query import make_query_run


class Faiss_QA:
    def __init__(self, llm, vectorstore, top_k=10):
        self.llm = llm
        self.vectorstore = vectorstore
        self.top_k = top_k

    async def run_faiss(self, query):
        """단일 쿼리에 대해 벡터 검색을 수행하고 컨텍스트를 반환"""
        try:
            # 벡터 검색 수행
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search, query, k=self.top_k
            )

            # 검색된 문서들을 컨텍스트로 결합
            contexts = []
            for doc in results:
                contexts.append(doc.page_content)

            return "\n\n".join(contexts)
        except Exception as e:
            print(f"벡터 검색 중 오류 발생: {str(e)}")
            return ""

    async def judge_llm(self, context, query):
        class Judge(BaseModel):
            result: Literal["정상", "이상"] = Field(
                description='찾은 내용을 바탕으로 이상치 판별. "정상" 또는 "이상"로 답변'
            )
            reason: Optional[str] = Field(
                default=None,
                description='"result"가 "이상"인 경우에만 이상하다고 판단한 이유를 찾은 내용을 바탕으로 자세히 설명. 반드시 한국어로 답변해.',
            )

        structured_llm = self.llm.with_structured_output(Judge)
        prompt = PromptTemplate.from_template(JUDGE_PROMPT)
        chain = prompt | structured_llm
        result = await asyncio.to_thread(
            chain.invoke, {"context": context, "query": query}
        )

        return result

    async def process_single_query(self, query_data):
        """단일 쿼리 데이터를 처리하여 판별 결과를 반환"""
        try:
            query = query_data["query"]
            metadata = query_data["metadata"]
            row_index = query_data["row_index"]

            # 1. 벡터 검색으로 컨텍스트 찾기
            context = await self.run_faiss(query)

            # 2. LLM으로 판별 수행
            judge_result = await self.judge_llm(context, query)

            return {
                "row_index": row_index,
                "metadata": metadata,
                "query": query,
                "context": context,
                "judge_result": judge_result,
                "success": True,
            }
        except Exception as e:
            return {
                "row_index": query_data.get("row_index", "unknown"),
                "error": str(e),
                "success": False,
            }

    async def process_all_queries(self, query_lst):
        """모든 쿼리를 비동기로 처리 (프로그레스바 포함)"""
        print(f"총 {len(query_lst)}개의 쿼리를 처리 시작...")

        # 모든 쿼리를 비동기로 처리 (프로그레스바와 함께)
        tasks = [self.process_single_query(query_data) for query_data in query_lst]

        # tqdm을 사용한 프로그레스바와 asyncio.gather 분리
        results = []
        with tqdm(total=len(tasks), desc="쿼리 처리 중", unit="쿼리") as pbar:
            # 작은 배치로 나누어서 처리 (메모리 효율성을 위해)
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
                pbar.update(len(batch))

        # 성공/실패 결과 분리
        successful_results = []
        failed_results = []

        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "success": False})
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)

        print(
            f"처리 완료: 성공 {len(successful_results)}개, 실패 {len(failed_results)}개"
        )

        return successful_results, failed_results

    def save_results_to_txt(
        self,
        successful_results,
        failed_results,
        filename="anomaly_detection_results.txt",
    ):
        """결과를 텍스트 파일로 저장 (이상치만 저장)"""
        print(f"결과를 {filename} 파일로 저장 중...")

        # 이상치만 필터링
        abnormal_results = [
            result
            for result in successful_results
            if result["judge_result"].result == "이상"
        ]

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== 이상치 판별 결과 (이상치만 표시) ===\n\n")
            f.write(f"총 처리된 쿼리: {len(successful_results)}개\n")
            f.write(f"정상 판별: {len(successful_results) - len(abnormal_results)}개\n")
            f.write(f"이상 판별: {len(abnormal_results)}개\n")
            f.write(f"실패한 쿼리: {len(failed_results)}개\n\n")

            if abnormal_results:
                # 이상치 결과들만 저장 (프로그레스바와 함께)
                f.write("=== 이상치 판별 결과 ===\n\n")

                for i, result in enumerate(
                    tqdm(abnormal_results, desc="이상치 결과 저장 중", unit="건"), 1
                ):
                    f.write(f"--- 이상치 {i} ---\n")
                    f.write(f"행 번호: {result['row_index']}\n")
                    f.write(f"구분: {result['metadata']['구분']}\n")
                    f.write(f"고지형식: {result['metadata']['고지형식']}\n")
                    f.write(f"세대유형: {result['metadata']['세대유형']}\n")
                    f.write(f"사용압력: {result['metadata']['사용(측정)압력']}\n")
                    f.write(f"등급: {result['metadata']['등급']}\n\n")

                    f.write(f"판별 결과: {result['judge_result'].result}\n")
                    if result["judge_result"].reason:
                        f.write(f"이상 이유: {result['judge_result'].reason}\n")

                    f.write(f"\n원본 쿼리:\n{result['query']}\n\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("=== 발견된 이상치가 없습니다 ===\n\n")

            # 실패한 결과들 저장
            if failed_results:
                f.write("\n=== 실패한 쿼리들 ===\n\n")
                for i, result in enumerate(failed_results, 1):
                    f.write(
                        f"실패 {i}: 행 번호 {result.get('row_index', 'unknown')} - {result.get('error', '알 수 없는 오류')}\n"
                    )

        print(f"결과가 {filename} 파일로 저장되었습니다.")
        print(f"총 {len(abnormal_results)}개의 이상치가 발견되었습니다.")

    def save_results_to_excel(
        self,
        successful_results,
        failed_results,
        original_file="merged_data.xlsx",
        output_file="anomaly_detection_final_results.xlsx",
    ):
        """결과를 기존 Excel 파일에 새로운 컬럼으로 추가하여 저장"""
        print(f"Excel 결과를 {output_file} 파일로 저장 중...")

        try:
            # 기존 Excel 파일 읽기
            df = pd.read_excel(original_file)
            print(f"기존 Excel 파일 로드 완료: {len(df)}행")

            # 새로운 컬럼 초기화
            df["이상치_판별결과"] = "미처리"
            df["이상치_판별이유"] = ""

            # 성공한 결과들을 데이터프레임에 매핑
            for result in tqdm(
                successful_results, desc="Excel 결과 매핑 중", unit="건"
            ):
                row_index = result["row_index"]
                judge_result = result["judge_result"].result
                reason = (
                    result["judge_result"].reason
                    if result["judge_result"].reason
                    else ""
                )

                # 해당 행에 결과 업데이트
                if row_index < len(df):
                    df.loc[row_index, "이상치_판별결과"] = judge_result
                    df.loc[row_index, "이상치_판별이유"] = reason

            # 실패한 결과들 처리
            for result in failed_results:
                row_index = result.get("row_index", None)
                if row_index is not None and row_index < len(df):
                    df.loc[row_index, "이상치_판별결과"] = "처리실패"
                    df.loc[row_index, "이상치_판별이유"] = result.get(
                        "error", "알 수 없는 오류"
                    )

            # 결과 Excel 파일로 저장
            df.to_excel(output_file, index=False)

            # 통계 정보 출력
            result_counts = df["이상치_판별결과"].value_counts()
            print(f"\nExcel 저장 완료: {output_file}")
            print("=== 최종 통계 ===")
            for result_type, count in result_counts.items():
                print(f"{result_type}: {count}건")

            return df

        except Exception as e:
            print(f"Excel 저장 중 오류 발생: {str(e)}")
            return None


# 전체 처리를 위한 메인 함수
async def main_process(query_lst, vectorstore, llm):
    """전체 처리 프로세스를 실행하는 메인 함수"""
    faiss_qa = Faiss_QA(llm, vectorstore)

    # 모든 쿼리 처리
    successful_results, failed_results = await faiss_qa.process_all_queries(query_lst)

    # TXT 파일로 결과 저장 (이상치만)
    faiss_qa.save_results_to_txt(successful_results, failed_results)

    # Excel 파일로 전체 결과 저장
    final_df = faiss_qa.save_results_to_excel(successful_results, failed_results)

    return successful_results, failed_results, final_df


if __name__ == "__main__":
    vectorstore, chunks, metadata = load_faiss_data()

    llm = initialize_llm("langchain_gpt4o")
    query_lst = make_query_run()
    successful_results, failed_results, final_df = asyncio.run(
        main_process(query_lst, vectorstore, llm)
    )
