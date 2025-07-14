import streamlit as st
import pandas as pd
import os
import tempfile
import asyncio
from datetime import datetime
import shutil

# 로컬 모듈 임포트
from _preprocess_exel import main as preprocess_main
from run import main_process
from embedding import load_faiss_data
from model import initialize_llm
from make_query import make_query

# 페이지 설정
st.set_page_config(
    page_title="가스 이상치 탐지 시스템",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 메인 타이틀
st.title("🔥 가스 이상치 탐지 시스템")
st.markdown("---")

# 사이드바
st.sidebar.header("📊 분석 단계")
st.sidebar.markdown(
    """
1. **Excel 파일 업로드**
2. **데이터 전처리**
3. **이상치 분석 실행**
4. **결과 다운로드**
"""
)

# 세션 상태 초기화
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "uploaded_file_processed" not in st.session_state:
    st.session_state.uploaded_file_processed = False

# 1. 파일 업로드 섹션
st.header("📁 1. Excel 파일 업로드")
uploaded_file = st.file_uploader(
    "분석할 Excel 파일을 업로드하세요",
    type=["xlsx", "xls"],
    help="data1.xlsx 형태의 파일을 업로드해주세요",
)

if uploaded_file is not None:
    # 업로드된 파일을 임시로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    # 파일을 data1.xlsx로 복사
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    target_path = os.path.join(data_dir, "data1.xlsx")
    shutil.copy2(temp_file_path, target_path)

    # 업로드된 파일 정보 표시
    try:
        df_preview = pd.read_excel(uploaded_file, sheet_name="학습데이터2(연소기정보)")
        st.success(f"✅ 파일이 성공적으로 업로드되었습니다!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 행 수", len(df_preview))
        with col2:
            st.metric("컬럼 수", len(df_preview.columns))

        # 데이터 미리보기
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(df_preview.head())

        st.session_state.uploaded_file_processed = True

    except Exception as e:
        st.error(f"❌ 파일 읽기 오류: {str(e)}")
        st.info("올바른 형식의 Excel 파일인지 확인해주세요.")

# 2. 데이터 전처리 섹션
st.header("⚙️ 2. 데이터 전처리")

if st.session_state.uploaded_file_processed:
    if st.button("🔄 전처리 시작", type="primary", use_container_width=True):
        with st.spinner("데이터 전처리 중... 잠시만 기다려주세요."):
            try:
                # 전처리 실행
                merge_path = preprocess_main()
                st.session_state.preprocessed = True
                st.session_state.merge_path = merge_path

                st.success("✅ 데이터 전처리가 완료되었습니다!")

                # 전처리된 데이터 정보 표시
                processed_df = pd.read_excel(merge_path)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("전처리 후 행 수", len(processed_df))
                with col2:
                    st.metric("컬럼 수", len(processed_df.columns))
                with col3:
                    unique_types = (
                        processed_df["고지형식"].nunique()
                        if "고지형식" in processed_df.columns
                        else 0
                    )
                    st.metric("고지형식 종류", unique_types)

                # 전처리된 데이터 미리보기
                with st.expander("📊 전처리된 데이터 미리보기"):
                    st.dataframe(processed_df.head())

            except Exception as e:
                st.error(f"❌ 전처리 중 오류 발생: {str(e)}")
else:
    st.info("먼저 Excel 파일을 업로드해주세요.")

# 3. 이상치 분석 섹션
st.header("🎯 3. 이상치 분석")

if st.session_state.preprocessed:
    st.info("💡 분석이 시작되면 시간이 오래 걸릴 수 있습니다. 브라우저를 닫지 마세요!")

    if st.button("🚀 이상치 분석 시작", type="primary", use_container_width=True):

        # 분석 진행 상황을 표시할 컨테이너들
        progress_container = st.container()
        log_container = st.container()

        with progress_container:
            st.markdown("### 📈 분석 진행 상황")
            progress_bar = st.progress(0)
            status_text = st.empty()

        with log_container:
            st.markdown("### 📝 실행 로그")
            log_placeholder = st.empty()

        try:
            # 분석 실행
            status_text.text("🔧 모델 및 벡터스토어 로딩 중...")
            progress_bar.progress(10)

            # 벡터스토어 로드
            vectorstore, chunks, metadata = load_faiss_data()
            progress_bar.progress(25)

            # LLM 초기화
            status_text.text("🤖 LLM 모델 초기화 중...")
            llm = initialize_llm("langchain_gpt4o")
            progress_bar.progress(40)

            # 쿼리 생성
            status_text.text("📋 분석 쿼리 생성 중...")
            query_lst = make_query("merged_data.xlsx")
            progress_bar.progress(55)

            status_text.text("🔍 이상치 분석 실행 중... (시간이 오래 걸릴 수 있습니다)")
            progress_bar.progress(60)

            # 비동기 분석 실행
            async def run_analysis():
                return await main_process(query_lst, vectorstore, llm)

            # 이벤트 루프에서 실행
            successful_results, failed_results, final_df = asyncio.run(run_analysis())

            progress_bar.progress(90)
            status_text.text("💾 결과 저장 중...")

            progress_bar.progress(100)
            status_text.text("✅ 분석이 완료되었습니다!")

            # 분석 결과 저장
            st.session_state.analysis_complete = True
            st.session_state.successful_results = successful_results
            st.session_state.failed_results = failed_results
            st.session_state.final_df = final_df

            # 결과 요약 표시
            st.success("🎉 이상치 분석이 완료되었습니다!")

            # 통계 정보
            abnormal_count = len(
                [r for r in successful_results if r["judge_result"].result == "이상"]
            )
            normal_count = len(successful_results) - abnormal_count

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 분석", len(query_lst), delta=None)
            with col2:
                st.metric("정상", normal_count, delta=None)
            with col3:
                st.metric(
                    "이상", abnormal_count, delta="⚠️" if abnormal_count > 0 else None
                )
            with col4:
                st.metric(
                    "실패",
                    len(failed_results),
                    delta="❌" if len(failed_results) > 0 else None,
                )

        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            progress_bar.progress(0)
            status_text.text("분석 실패")

else:
    st.info("먼저 데이터 전처리를 완료해주세요.")

# 4. 결과 다운로드 섹션
st.header("📥 4. 결과 다운로드")

if st.session_state.analysis_complete:
    st.success("분석이 완료되어 결과를 다운로드할 수 있습니다!")

    col1, col2 = st.columns(2)

    with col1:
        # Excel 결과 다운로드
        if os.path.exists("anomaly_detection_final_results.xlsx"):
            with open("anomaly_detection_final_results.xlsx", "rb") as file:
                st.download_button(
                    label="📊 Excel 결과 다운로드",
                    data=file.read(),
                    file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    with col2:
        # 텍스트 결과 다운로드
        if os.path.exists("anomaly_detection_results.txt"):
            with open("anomaly_detection_results.txt", "rb") as file:
                st.download_button(
                    label="📝 텍스트 결과 다운로드",
                    data=file.read(),
                    file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # 이상치만 표시
    if "successful_results" in st.session_state:
        abnormal_results = [
            r
            for r in st.session_state.successful_results
            if r["judge_result"].result == "이상"
        ]

        if abnormal_results:
            st.markdown("### 🚨 발견된 이상치")

            for i, result in enumerate(abnormal_results[:5], 1):  # 처음 5개만 표시
                with st.expander(
                    f"이상치 {i}: 행 {result['row_index']} (구분: {result['metadata']['구분']})"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**기본 정보**")
                        st.write(f"- 고지형식: {result['metadata']['고지형식']}")
                        st.write(f"- 세대유형: {result['metadata']['세대유형']}")
                        st.write(f"- 사용압력: {result['metadata']['사용(측정)압력']}")
                        st.write(f"- 등급: {result['metadata']['등급']}")

                    with col2:
                        st.write("**판별 결과**")
                        st.write(f"- 결과: **{result['judge_result'].result}**")
                        if result["judge_result"].reason:
                            st.write(f"- 이유: {result['judge_result'].reason}")

            if len(abnormal_results) > 5:
                st.info(
                    f"총 {len(abnormal_results)}개의 이상치가 발견되었습니다. 전체 결과는 다운로드 파일에서 확인하세요."
                )
        else:
            st.info("🎉 이상치가 발견되지 않았습니다!")

else:
    st.info("분석을 완료한 후 결과를 다운로드할 수 있습니다.")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🔥 가스 이상치 탐지 시스템 v1.0 | 
        Built with Streamlit & LangChain
    </div>
    """,
    unsafe_allow_html=True,
)
