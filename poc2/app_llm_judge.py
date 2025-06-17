# streamlit_app.py
import streamlit as st
import asyncio
from _llm_judge import Analyze  # 너가 작성한 클래스
from utils import get_data_from_txt
from _preprocess import preprocess_excel, excel_to_txt
from model import initialize_llm

st.title("📊 LLM 자체 판단 이상치 탐지")


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
                results = await analyzer.run_data_lst(st.session_state.data_lst)
                st.success("분석 완료 되어 보고서를 작성 중입니다.")
                report = analyzer.reports_llm(results)
                return report

        final_report = asyncio.run(run_analysis())

        st.subheader("📝 최종 리포트")
        st.markdown(final_report)
