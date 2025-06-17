# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from _llm_judge_std import Analyze, save_results_to_txt  # 너가 작성한 클래스
from utils import get_data_from_txt
from _preprocess import preprocess_excel, excel_to_txt
from model import initialize_llm

# 페이지 설정
st.set_page_config(layout="wide", page_title="에너지 사용량 분석 도구")

# 메인 타이틀
st.title("📊 평균/표준편차를 고려한 이상치 탐지")


# 데이터 파일 처리
data_file_path = "./data2_biz_with_std.xlsx"

uploaded_file = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])


# 이상치 분석 데이터 처리
if "data_lst" not in st.session_state:
    st.session_state.data_lst = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# 파일 전처리 (이상치 분석용)
if uploaded_file is not None and st.session_state.data_lst is None:
    with st.spinner("📁 파일 처리 중..."):
        input_path = "./uploaded.xlsx"
        output_path = "./data2_preprocessed.xlsx"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        preprocess_excel(input_path, output_path)
        get_txt = excel_to_txt(output_path)
        st.session_state.data_lst = get_data_from_txt(get_txt)

        st.success("✅ 파일 전처리 완료")

try:
    df = pd.read_excel(data_file_path)

    # 그룹명 추정 (첫 번째 컬럼)
    group_col = df.columns[0]

    # 월별 컬럼 리스트
    month_labels = [f"{i}월" for i in range(1, 13)]
    month_cols = [f"사용량_{month}_평균" for month in month_labels]

    # 모든 그룹 선택
    unique_groups = df[group_col].unique().tolist()
    selected_groups = unique_groups

    # === 왼쪽: 그룹별 사용량 요약 ===
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 그룹별 월별 사용량 평균/표준편차")

        for group in selected_groups:
            with st.expander(f"🔹 {group}", expanded=False):
                row = df[df[group_col] == group].iloc[0]

                # 열량 정보 컴팩트하게 표시
                st.write("**열량 정보**")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("보일러 평균", f"{row['보일러_열량_평균']:.1f}")
                    st.metric("연소기 평균", f"{row['연소기_열량_평균']:.1f}")
                with metric_col2:
                    st.metric("보일러 표준편차", f"{row['보일러_열량_표준편차']:.1f}")
                    st.metric("연소기 표준편차", f"{row['연소기_열량_표준편차']:.1f}")

                # 월별 사용량 요약 (컴팩트)
                month_data = []
                for i in range(1, 13):
                    month = f"{i}월"
                    avg_col = f"사용량_{month}_평균"
                    std_col = f"사용량_{month}_표준편차"
                    if avg_col in df.columns and std_col in df.columns:
                        month_data.append(
                            [
                                month,
                                round(row[avg_col], 1),
                                round(row[std_col], 1),
                            ]
                        )

                month_df = pd.DataFrame(month_data, columns=["월", "평균", "표준편차"])
                st.dataframe(month_df.set_index("월"), height=200)
        # === 하단: 통합 그래프 ===
        st.markdown("---")
        st.subheader("📈 통합 시각화")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            show_usage_plot = st.button(
                "📊 사용량 추이", type="primary", use_container_width=True
            )
        with graph_col2:
            show_std_plot = st.button("📈 표준편차 추이", use_container_width=True)

        # 그래프 표시
        if show_usage_plot:
            st.subheader("📊 선택된 그룹의 월별 평균 사용량 추이")

            plt.rcParams["font.family"] = [
                "AppleGothic",
            ]
            plt.rcParams["axes.unicode_minus"] = False

            fig, ax = plt.subplots(figsize=(14, 8))

            colors = plt.cm.Set3(range(len(selected_groups)))

            for idx, group in enumerate(selected_groups):
                group_row = df[df[group_col] == group]
                if not group_row.empty:
                    usage_values = group_row[month_cols].values.flatten()
                    ax.plot(
                        month_labels,
                        usage_values,
                        label=group,
                        marker="o",
                        linewidth=2.5,
                        color=colors[idx],
                        markersize=6,
                    )

            ax.set_title("그룹별 월별 평균 사용량 비교", fontsize=18, pad=20)
            ax.set_xlabel("월", fontsize=14)
            ax.set_ylabel("평균 사용량", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close()

        if show_std_plot:
            st.subheader("📈 선택된 그룹의 월별 사용량 표준편차")

            plt.rcParams["font.family"] = [
                "AppleGothic",
            ]
            plt.rcParams["axes.unicode_minus"] = False

            fig, ax = plt.subplots(figsize=(14, 8))

            std_cols = [f"사용량_{month}_표준편차" for month in month_labels]

            for group in selected_groups:
                group_row = df[df[group_col] == group]
                if not group_row.empty:
                    std_values = group_row[std_cols].values.flatten()
                    ax.plot(
                        month_labels,
                        std_values,
                        label=group,
                        marker="s",
                        linewidth=2.5,
                    )

            ax.set_title("그룹별 월별 사용량 표준편차 비교", fontsize=18, pad=20)
            ax.set_xlabel("월", fontsize=14)
            ax.set_ylabel("표준편차", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close()

    with col_right:
        st.subheader("🔍 이상치 분석 결과")

        # 이상치 분석 시작 버튼
        if st.session_state.data_lst is not None:
            if st.session_state.analysis_results is None:
                if st.button(
                    "🚀 이상치 분석 시작",
                    type="primary",
                    key="integrated_analysis",
                ):

                    async def run_analysis():
                        with st.spinner("🤖 이상치 분석 중..."):
                            llm = initialize_llm("langchain_gpt4o")
                            analyzer = Analyze(llm)

                            # 진행률 표시
                            total = len(st.session_state.data_lst)
                            progress_bar = st.progress(0, text="🔍 분석 진행 중...")
                            results = []

                            semaphore = asyncio.Semaphore(30)
                            processed_count = 0

                            async def process_with_progress(data_item):
                                nonlocal processed_count
                                async with semaphore:
                                    result = await analyzer.process_single_item(
                                        data_item
                                    )
                                    processed_count += 1
                                    progress_bar.progress(
                                        processed_count / total,
                                        text=f"{processed_count} / {total} 완료",
                                    )
                                    return result

                            tasks = [
                                process_with_progress(item)
                                for item in st.session_state.data_lst
                            ]
                            results = await asyncio.gather(*tasks)

                            progress_bar.progress(1.0, text="✅ 분석 완료")
                            outlier_data = [
                                {"category": item["gt_data"]["그룹"], **item}
                                for item in results
                                if item["judge_result"].result == "이상"
                            ]
                            st.success("최종 보고서를 작성 중입니다.")
                            final_ans = analyzer.reports_llm(outlier_data)

                            # 결과를 파일로 저장
                            # output_path = "./integrated_analysis_results.txt"
                            # saved_path = save_results_to_txt(output_path, results)

                            return final_ans

                    try:
                        final_ans = asyncio.run(run_analysis())
                        st.session_state.analysis_results = final_ans
                        st.success("🎉 이상치 분석 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류: {str(e)}")

            else:
                # 분석 결과 표시 - analysis_results가 문자열(리포트 내용)인 경우
                if isinstance(st.session_state.analysis_results, str):
                    st.success("🎉 이상치 분석 완료!")
                    st.subheader("📝 최종 리포트")
                    st.markdown(st.session_state.analysis_results)

                # 분석 결과가 파일 경로인 경우
                else:
                    try:
                        txt_path = st.session_state.analysis_results
                        st.subheader("📄 분석 결과 요약")
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.code(content, language="json")
                    except (FileNotFoundError, TypeError, AttributeError) as e:
                        st.error(f"❌ 결과 파일을 읽을 수 없습니다: {str(e)}")
                        # 오류 발생 시 analysis_results 초기화
                        st.session_state.analysis_results = None
                        st.info("🔄 분석을 다시 시작해주세요.")


except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")
