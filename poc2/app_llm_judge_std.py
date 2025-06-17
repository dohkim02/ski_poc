# streamlit_app.py
import streamlit as st
import asyncio
from _llm_judge_std import Analyze, save_results_to_txt  # 너가 작성한 클래스
from utils import get_data_from_txt
from _preprocess import preprocess_excel, excel_to_txt
from model import initialize_llm

st.title("📊 월별 에너지 사용량 이상치")


# 스트림릿 progress bar 사용 예
async def run_with_progress(analyzer, data_lst):
    total = len(data_lst)
    progress_bar = st.progress(0, text="🔍 분석 진행 중...")
    results = []

    semaphore = asyncio.Semaphore(30)

    processed_count = 0  # 상태 추적용

    async def process_with_progress(data_item):
        nonlocal processed_count
        async with semaphore:
            result = await analyzer.process_single_item(data_item)
            processed_count += 1
            progress_bar.progress(
                processed_count / total, text=f"{processed_count} / {total} 완료"
            )
            return result

    tasks = [process_with_progress(item) for item in data_lst]
    results = await asyncio.gather(*tasks)

    progress_bar.progress(1.0, text="✅ 분석 완료")
    return results


# 1. 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])

# 2. 파일 처리 완료 상태 플래그
if "data_lst" not in st.session_state:
    st.session_state.data_lst = None

# 3. 파일 전처리
if uploaded_file is not None and st.session_state.data_lst is None:
    with st.spinner("파일 처리 중..."):
        input_path = "./uploaded.xlsx"
        output_path = "./data2_preprocessed.xlsx"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        preprocess_excel(input_path, output_path)
        get_txt = excel_to_txt(output_path)
        st.session_state.data_lst = get_data_from_txt(get_txt)

        st.success("파일 전처리 완료 ✅")

# 4. 분석 버튼
if st.session_state.data_lst is not None:
    if st.button("📈 분석 시작"):

        async def run_analysis():
            with st.spinner("LLM 분석 중..."):
                llm = initialize_llm("langchain_gpt4o")
                analyzer = Analyze(llm)

                # ✅ 진행률 포함된 함수 호출
                results = await run_with_progress(analyzer, st.session_state.data_lst)

                st.success("분석 완료")
                output_path = "./temp(llm_std).txt"
                saved_path = save_results_to_txt(output_path, results)
                return saved_path

        txt_path = asyncio.run(run_analysis())

        # 결과 표시
        st.subheader("📄 분석 결과 요약")
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.code(content, language="json")
