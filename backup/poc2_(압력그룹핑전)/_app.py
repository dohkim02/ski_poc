import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# 필요한 모듈 import
from _run import Analyze, main as run_main
from _preprocess import excel_to_txt, preprocess_excel
from utils import get_data_from_txt, get_previous_monthes
import ast

# 모델 경로 설정
MODEL_PATH = os.path.abspath("../")
sys.path.append(MODEL_PATH)
from model import initialize_llm

# Streamlit 페이지 설정
st.set_page_config(page_title="이상치 분석 시스템", page_icon="📊", layout="wide")

# 세션 상태 초기화
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "results" not in st.session_state:
    st.session_state.results = None
if "post_processing_results" not in st.session_state:
    st.session_state.post_processing_results = None


def create_comparison_chart(item, case_num, colors):
    """기준값과 비교값을 시각화하는 차트 생성"""
    standard_data = item["standard_data"]
    comparison_data = item["comparison_input_data"]

    # 월 순서 정의
    month_order = [
        "1월",
        "2월",
        "3월",
        "4월",
        "5월",
        "6월",
        "7월",
        "8월",
        "9월",
        "10월",
        "11월",
        "12월",
    ]

    # 공통 월만 추출하고 순서대로 정렬
    common_months = list(set(standard_data.keys()) & set(comparison_data.keys()))
    common_months = [month for month in month_order if month in common_months]

    if not common_months:
        return None

    # 데이터 준비
    standard_values = [standard_data[month] for month in common_months]
    comparison_values = [comparison_data[month] for month in common_months]

    # 색상 설정
    color_pair = colors[case_num % len(colors)]

    fig = go.Figure()

    # 기준값 라인
    fig.add_trace(
        go.Scatter(
            x=common_months,
            y=standard_values,
            mode="lines+markers",
            name=f"기준값 (케이스 {case_num})",
            line=dict(color=color_pair[0], width=3),
            marker=dict(size=8),
        )
    )

    # 비교값 라인
    fig.add_trace(
        go.Scatter(
            x=common_months,
            y=comparison_values,
            mode="lines+markers",
            name=f"비교값 (케이스 {case_num})",
            line=dict(color=color_pair[1], width=3, dash="dash"),
            marker=dict(size=8),
        )
    )

    # 레이아웃 설정
    fig.update_layout(
        title=f"케이스 {case_num}: 기준값 vs 비교값",
        xaxis_title="월",
        yaxis_title="값",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def create_all_cases_chart(filtered_results):
    """모든 케이스를 한 번에 보여주는 차트"""
    fig = make_subplots(
        rows=len(filtered_results),
        cols=1,
        subplot_titles=[f"케이스 {i+1}" for i in range(len(filtered_results))],
        vertical_spacing=0.08,
    )

    # 색상 팔레트
    colors = px.colors.qualitative.Set3

    month_order = [
        "1월",
        "2월",
        "3월",
        "4월",
        "5월",
        "6월",
        "7월",
        "8월",
        "9월",
        "10월",
        "11월",
        "12월",
    ]

    for i, item in enumerate(filtered_results):
        standard_data = item["standard_data"]
        comparison_data = item["comparison_input_data"]

        common_months = list(set(standard_data.keys()) & set(comparison_data.keys()))
        common_months = [month for month in month_order if month in common_months]

        if common_months:
            standard_values = [standard_data[month] for month in common_months]
            comparison_values = [comparison_data[month] for month in common_months]

            # 기준값
            fig.add_trace(
                go.Scatter(
                    x=common_months,
                    y=standard_values,
                    mode="lines+markers",
                    name=f"기준값 {i+1}",
                    line=dict(color=colors[i * 2 % len(colors)], width=2),
                    marker=dict(size=6),
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

            # 비교값
            fig.add_trace(
                go.Scatter(
                    x=common_months,
                    y=comparison_values,
                    mode="lines+markers",
                    name=f"비교값 {i+1}",
                    line=dict(
                        color=colors[(i * 2 + 1) % len(colors)], width=2, dash="dash"
                    ),
                    marker=dict(size=6),
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=400 * len(filtered_results),
        title_text="전체 케이스 비교",
        showlegend=True,
    )

    return fig


async def run_analysis(data_file_path):
    """분석 실행"""
    try:
        # LLM 초기화
        with st.spinner("LLM 모델을 초기화하는 중..."):
            llm = initialize_llm("langchain_gpt4o")

        # 데이터 로드
        with st.spinner("데이터를 로드하는 중..."):
            data_lst = get_data_from_txt(data_file_path)

        st.success(f"총 {len(data_lst)}개의 데이터를 로드했습니다.")

        analyzer = Analyze(llm)

        # 1차 분석 실행
        st.info("🔍 1차 분석을 시작합니다...")

        # 프로그레스 바와 상태 표시를 위한 컨테이너
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # 실시간 프로그레스 업데이트를 위한 커스텀 함수
        async def run_biz_judge_with_progress(data_lst):
            """프로그레스 바와 함께 1차 분석 실행"""
            results = []
            total = len(data_lst)

            # 세마포어로 동시 실행 수 제한
            semaphore = asyncio.Semaphore(50)

            async def process_with_streamlit_progress(data_item, index):
                async with semaphore:
                    result = await analyzer.process_single_item(data_item)
                    # Streamlit 프로그레스 바 업데이트
                    progress = (index + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"처리 중: {index + 1}/{total} ({progress:.1%})")
                    return result

            # 모든 작업을 비동기로 실행
            tasks = [
                process_with_streamlit_progress(data_item, i)
                for i, data_item in enumerate(data_lst)
            ]
            results = await asyncio.gather(*tasks)

            return results

        results = await run_biz_judge_with_progress(data_lst)

        # 완료 상태 업데이트
        status_text.text(f"✅ 1차 분석 완료: {len(data_lst)}개 처리됨")

        # 이상치 필터링
        outlier_results = [item for item in results if item["judge_result"]]

        st.success(
            f"1차 분석 완료: {len(outlier_results)}개의 이상치가 발견되었습니다."
        )

        # 2차 패턴 체크
        if outlier_results:
            st.info("🔍 2차 패턴 체크를 시작합니다...")

            # 2차 분석용 프로그레스 바
            progress_container2 = st.container()
            with progress_container2:
                progress_bar2 = st.progress(0)
                status_text2 = st.empty()

            async def run_pattern_check_with_progress(outlier_results):
                """프로그레스 바와 함께 2차 패턴 체크 실행"""
                results = []
                total = len(outlier_results)

                # 세마포어로 동시 실행 수 제한
                semaphore = asyncio.Semaphore(50)

                async def process_pattern_check_with_progress(outlier_item, index):
                    async with semaphore:
                        latest_6_month_data = outlier_item["comparison_input_data"]
                        years_data = outlier_item["input_data"]["3년치 데이터"]
                        rest_month_data = get_previous_monthes(years_data)
                        pattern_result = await analyzer.pattern_checker(
                            rest_month_data, latest_6_month_data
                        )

                        # Streamlit 프로그레스 바 업데이트
                        progress = (index + 1) / total
                        progress_bar2.progress(progress)
                        status_text2.text(
                            f"패턴 체크 중: {index + 1}/{total} ({progress:.1%})"
                        )

                        # 기존 outlier_item에 pattern_result 추가
                        result_item = outlier_item.copy()
                        result_item["pattern_result"] = pattern_result
                        return result_item

                # 모든 작업을 비동기로 실행
                tasks = [
                    process_pattern_check_with_progress(outlier_item, i)
                    for i, outlier_item in enumerate(outlier_results)
                ]
                results = await asyncio.gather(*tasks)

                return results

            outlier_results = await run_pattern_check_with_progress(outlier_results)

            # 완료 상태 업데이트
            status_text2.text(f"✅ 2차 패턴 체크 완료: {len(outlier_results)}개 처리됨")
            st.success("2차 패턴 체크 완료!")

        return results, outlier_results

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        return None, None


def main():
    st.title("📊 이상치 분석 시스템")
    st.markdown("---")

    # 사이드바
    st.sidebar.title("📋 메뉴")

    # 파일 업로드
    st.sidebar.subheader("1. 데이터 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "Excel 파일을 업로드하세요",
        type=["xlsx", "xls"],
        help="분석할 데이터가 포함된 Excel 파일을 선택하세요",
    )

    if uploaded_file is not None:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_excel_path = tmp_file.name

        # 전처리
        st.sidebar.subheader("2. 전처리")
        if st.sidebar.button("전처리 실행"):
            with st.spinner("데이터를 전처리하는 중..."):
                try:
                    # 1단계: preprocess_excel 실행 (Excel → 전처리된 Excel)
                    preprocessed_excel_path = "./preprocessed.xlsx"
                    preprocess_excel(temp_excel_path, preprocessed_excel_path)

                    # 2단계: excel_to_txt 실행 (전처리된 Excel → TXT)
                    preprocessed_path = excel_to_txt(
                        preprocessed_excel_path, "./preprocessed.txt"
                    )
                    st.sidebar.success("전처리 완료!")
                    st.session_state.preprocessed_path = preprocessed_path
                except Exception as e:
                    st.sidebar.error(f"전처리 중 오류: {str(e)}")

        # 분석 실행
        st.sidebar.subheader("3. 분석 실행")
        if hasattr(st.session_state, "preprocessed_path") and st.sidebar.button(
            "분석 시작"
        ):
            results, outlier_results = asyncio.run(
                run_analysis(st.session_state.preprocessed_path)
            )

            if results is not None:
                st.session_state.results = results
                st.session_state.outlier_results = outlier_results
                st.session_state.analysis_complete = True

                # 후처리 결과 필터링
                if outlier_results and "pattern_result" in outlier_results[0]:
                    filtered_results = []
                    for item in outlier_results:
                        pattern_result = item["pattern_result"]
                        result_value = getattr(pattern_result, "result", None)
                        if result_value is None and isinstance(pattern_result, dict):
                            result_value = pattern_result.get("result")
                        if result_value == "yes":
                            filtered_results.append(item)

                    st.session_state.post_processing_results = filtered_results

    # 메인 컨텐츠 영역
    if st.session_state.analysis_complete:
        st.subheader("📈 분석 결과")

        # 기본 통계
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("전체 데이터", len(st.session_state.results))

        with col2:
            st.metric("1차 이상치", len(st.session_state.outlier_results))

        with col3:
            if st.session_state.post_processing_results:
                st.metric("최종 이상치", len(st.session_state.post_processing_results))
            else:
                st.metric("최종 이상치", 0)

        # 후처리 결과 시각화
        if st.session_state.post_processing_results:
            st.subheader("🎯 최종 이상치 분석 결과")

            filtered_results = st.session_state.post_processing_results

            # 시각화 옵션
            viz_option = st.radio(
                "시각화 옵션 선택:",
                ["개별 차트", "전체 차트", "상세 정보"],
                horizontal=True,
            )

            # 색상 팔레트 정의
            color_pairs = [
                ("#1f77b4", "#ff7f0e"),  # 파란색-주황색
                ("#2ca02c", "#d62728"),  # 초록색-빨간색
                ("#9467bd", "#8c564b"),  # 보라색-갈색
                ("#e377c2", "#7f7f7f"),  # 분홍색-회색
                ("#bcbd22", "#17becf"),  # 올리브-청록색
                ("#aec7e8", "#ffbb78"),  # 연한 파란색-연한 주황색
            ]

            if viz_option == "개별 차트":
                for i, item in enumerate(filtered_results):
                    with st.expander(
                        f"케이스 {i+1} - 구분: {item['input_data']['구분']} {item['ground_truth']['category']}",
                        expanded=True,
                    ):
                        # 패턴 결과 정보
                        pattern_result = item["pattern_result"]
                        result_value = getattr(pattern_result, "result", None)
                        reason_value = getattr(pattern_result, "reason", None)
                        if result_value is None and isinstance(pattern_result, dict):
                            result_value = pattern_result.get("result")
                            reason_value = pattern_result.get("reason")

                        st.write(f"**판정 결과:** {result_value}")
                        st.write(f"**이유:** {reason_value}")

                        # 차트 생성
                        fig = create_comparison_chart(item, i + 1, color_pairs)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        # 데이터 테이블
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**기준 데이터:**")
                            st.json(item["standard_data"])
                        with col2:
                            st.write("**비교 데이터:**")
                            st.json(item["comparison_input_data"])

            elif viz_option == "전체 차트":
                st.write("모든 케이스를 한 번에 보여주는 차트입니다.")
                fig = create_all_cases_chart(filtered_results)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "상세 정보":
                # 데이터프레임으로 표시
                summary_data = []
                for i, item in enumerate(filtered_results):
                    pattern_result = item["pattern_result"]
                    result_value = getattr(pattern_result, "result", None)
                    reason_value = getattr(pattern_result, "reason", None)
                    if result_value is None and isinstance(pattern_result, dict):
                        result_value = pattern_result.get("result")
                        reason_value = pattern_result.get("reason")

                    summary_data.append(
                        {
                            "케이스": i + 1,
                            "카테고리": item["ground_truth"]["category"],
                            "판정": result_value,
                            "데이터 개수": item["ground_truth"]["data_num"],
                            "이유": (
                                reason_value[:100] + "..."
                                if reason_value and len(reason_value) > 100
                                else reason_value
                            ),
                        }
                    )

                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)

        else:
            st.info("최종 이상치가 발견되지 않았습니다.")

    else:
        # 초기 화면
        st.markdown(
            """
        ## 🚀 시작하기
        
        1. **왼쪽 사이드바**에서 Excel 파일을 업로드하세요
        2. **전처리 실행** 버튼을 클릭하세요
        3. **분석 시작** 버튼을 클릭하여 이상치 분석을 실행하세요
        
        ### 📋 기능
        - **1차 분석**: 기준 데이터 대비 이상치 탐지
        - **2차 패턴 체크**: AI를 통한 패턴 분석
        - **시각화**: 기준값 vs 비교값 그래프
        - **상세 정보**: 케이스별 분석 결과
        """
        )

        # 샘플 차트 보여주기
        st.subheader("📊 시각화 예시")

        # 샘플 데이터로 차트 생성
        sample_months = ["1월", "2월", "3월", "4월", "5월", "6월"]
        sample_standard = [100, 120, 110, 130, 115, 125]
        sample_comparison = [80, 95, 85, 105, 90, 100]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sample_months,
                y=sample_standard,
                mode="lines+markers",
                name="기준값",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sample_months,
                y=sample_comparison,
                mode="lines+markers",
                name="비교값",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            title="샘플 차트: 기준값 vs 비교값",
            xaxis_title="월",
            yaxis_title="값",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
