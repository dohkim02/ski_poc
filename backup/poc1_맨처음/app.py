import streamlit as st
import os
import asyncio
from _analyze import *
from _qna import *
from app_utils import *
import concurrent.futures
import threading
import time

# ✅ Streamlit 페이지 기본 설정
st.set_page_config(page_title="[과제1] 이상치 데이터 분석 & 질의 응답")
st.title("[과제1] 이상치 데이터 분석 및 질의 응답")

# ✅ 페이지 너비 조정 (선택사항)
st.markdown(
    """
<style>
.main .block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ✅ 모델 초기화 (1회만)
if "llm" not in st.session_state:
    st.session_state["llm"] = initialize_llm("gpt4o")

llm = st.session_state["llm"]

# ✅ 파일 업로드 처리
uploaded_file = st.file_uploader("TXT 파일을 업로드하세요", type=["txt"])

# ✅ 파일이 업로드되고 아직 분석하지 않은 경우에만 분석 실행
if uploaded_file is not None and "analysis_done" not in st.session_state:
    # 업로드된 파일 저장
    with open("uploaded_data.txt", "wb") as f:
        f.write(uploaded_file.read())
    st.success("파일 업로드 완료! 분석을 시작합니다...")

    # ✅ 실제 진행률을 추적하여 분석하는 함수
    def analyze_with_real_progress():
        all_data_chunks = get_data_from_txt("uploaded_data.txt")
        result_file = "./final_results.jsonl"

        # ✅ 진행률 추적기 생성
        progress_tracker = ProgressTracker()
        analysis_result = {"completed": False, "count": 0, "error": None}

        def run_analysis():
            try:
                asyncio.run(
                    batch_process_with_tracker(
                        llm, all_data_chunks, result_file, 20, progress_tracker
                    )
                )
                count_line = count_jsonl_lines_simple(result_file)
                analysis_result["count"] = count_line
                analysis_result["completed"] = True
            except Exception as e:
                analysis_result["error"] = str(e)
                analysis_result["completed"] = True

        # ✅ 분석을 별도 스레드에서 실행
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()

        return analysis_thread, progress_tracker, analysis_result

    # ✅ 진행률 표시 UI 생성
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("분석 중..."):
        analysis_thread, progress_tracker, analysis_result = (
            analyze_with_real_progress()
        )

        # ✅ 진행률 모니터링
        while analysis_thread.is_alive():
            progress_data = progress_tracker.get_progress()
            if progress_data:
                completed, total = progress_data
                progress = completed / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(
                    f"분석 진행 중... {completed}/{total} ({progress*100:.1f}%)"
                )

            time.sleep(0.1)  # 100ms마다 체크

        # ✅ 완료 처리
        analysis_thread.join()

        if analysis_result["error"]:
            st.error(f"분석 중 오류 발생: {analysis_result['error']}")
        else:
            progress_bar.progress(1.0)
            count_line = analysis_result["count"]
            status_text.text(
                f"분석 완료! 총 {count_line}개의 이상치 데이터를 찾았습니다."
            )

            st.success("모든 결과가 'final_results.jsonl'에 저장되었습니다!")
            st.write(f"이상치라고 판단한 데이터의 총 개수: {count_line}")

            # ✅ 분석 완료 상태 저장
            st.session_state["analysis_done"] = True
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["total_count"] = count_line

# ✅ 분석 완료 상태 표시
if st.session_state.get("analysis_done", False):
    st.info(
        f"📊 분석 완료된 파일: {st.session_state.get('uploaded_filename', 'unknown')}"
    )
    st.info(f"🔢 이상치 데이터 총 개수: {st.session_state.get('total_count', 0)}")


# ✅ 사용자 질의 처리 (분석이 완료된 경우에만)
if st.session_state.get("analysis_done", False):
    st.markdown("---")
    st.subheader("💬 질의응답")

    query = st.text_input("🔍 질문을 입력하세요:")

    # ✅ 새로운 질문이 입력되었는지 확인
    if query and (query != st.session_state.get("last_query", "")):
        st.write("질문을 처리 중입니다...")

        # ✅ 비동기 질의 처리도 ThreadPoolExecutor로 안전하게 실행
        def process_query():
            result_file = "./final_results.jsonl"
            chunks = load_jsonl_chunks(result_file, chunk_size=140)
            results = asyncio.run(batch_llm_answers(llm, query, chunks))
            return [r for r in results if r is not None]

        with st.spinner("답변 생성 중..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_query)
                all_results = future.result()

        # ✅ 결과를 세션 상태에 저장
        st.session_state["query_results"] = all_results
        st.session_state["last_query"] = query
        st.session_state["page"] = 0  # 새 질문 시 첫 페이지로 리셋

    # ✅ 저장된 결과가 있으면 페이지네이션으로 표시
    if "query_results" in st.session_state and st.session_state["query_results"]:
        all_results = st.session_state["query_results"]

        # ✅ 10개씩 페이지로 출력
        chunk_size = 10
        num_chunks = (len(all_results) + chunk_size - 1) // chunk_size

        if "page" not in st.session_state:
            st.session_state["page"] = 0

        page = st.session_state["page"]
        start = page * chunk_size
        end = start + chunk_size

        st.markdown(f"**질문: {st.session_state.get('last_query', '')}**")
        st.markdown(
            f"**총 {len(all_results)}개의 결과 중 {start+1}-{min(end, len(all_results))}번째 결과:**"
        )

        for i, answer in enumerate(all_results[start:end], start + 1):
            with st.expander(f"결과 {i}"):
                st.markdown(answer)

        # ✅ 페이지네이션 버튼 (rerun 없이 상태만 변경)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if page > 0:
                if st.button("⬅️ 이전"):
                    st.session_state["page"] -= 1

        with col2:
            st.write(f"페이지 {page + 1} / {num_chunks}")

        with col3:
            if end < len(all_results):
                if st.button("다음 ➡️"):
                    st.session_state["page"] += 1

else:
    st.write("👆 TXT 파일을 업로드하여 분석을 시작하세요.")
