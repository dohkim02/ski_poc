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
from datetime import datetime

# 필요한 모듈 import
from _run import Analyze, main
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


def create_3years_chart(item, case_num, colors):
    """3년치 데이터와 표준값을 연속으로 시각화하는 차트 생성"""
    try:
        # _run.py에서 사용하는 데이터 구조로 접근
        years_data = item["input_data"]["3년치 데이터"]
        standard = item["ground_truth"]["standard"]

        # 문자열인 경우 파싱
        if isinstance(years_data, str):
            years_data = ast.literal_eval(years_data)

        # 월 매핑 (숫자 -> 한글)
        month_mapping = {
            "01": "1월",
            "02": "2월",
            "03": "3월",
            "04": "4월",
            "05": "5월",
            "06": "6월",
            "07": "7월",
            "08": "8월",
            "09": "9월",
            "10": "10월",
            "11": "11월",
            "12": "12월",
        }

        # 년도 정렬
        sorted_years = sorted(years_data.keys())

        # 연속된 시간축과 데이터 생성
        x_labels = []
        actual_values = []
        standard_values = []

        # 년도별 색상 정의
        year_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for year in sorted_years:
            year_data = years_data[year]
            # 월을 01~12 순서로 정렬
            sorted_months = sorted(year_data.keys())

            for month in sorted_months:
                # x축 라벨 생성 (예: '22-04')
                x_labels.append(f"{year}-{month}")
                # 실제 데이터값
                actual_values.append(year_data[month])
                # 표준값 (월 매핑 후)
                month_korean = month_mapping.get(month, f"{int(month)}월")
                standard_val = standard.get(month_korean, 0)
                standard_values.append(standard_val)

        fig = go.Figure()

        # 표준값 라인 (회색, 점선)
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=standard_values,
                mode="lines+markers",
                name="표준값 (반복)",
                line=dict(color="#888888", width=2, dash="dot"),
                marker=dict(size=6),
                opacity=0.7,
            )
        )

        # 년도별로 실제 데이터 추가
        for i, year in enumerate(sorted_years):
            year_data = years_data[year]
            sorted_months = sorted(year_data.keys())

            # 해당 년도의 x축 라벨과 값들
            year_x_labels = [f"{year}-{month}" for month in sorted_months]
            year_values = [year_data[month] for month in sorted_months]

            # 년도별 색상
            color = year_colors[i % len(year_colors)]

            fig.add_trace(
                go.Scatter(
                    x=year_x_labels,
                    y=year_values,
                    mode="lines+markers",
                    name=f"20{year}년 실제값",
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                )
            )

        # 레이아웃 설정
        fig.update_layout(
            title=f"케이스 {case_num}: 3년치 데이터 vs 표준값 비교",
            xaxis_title="시기 (년-월)",
            yaxis_title="값",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            xaxis=dict(
                tickangle=45,
                tickmode="array",
                tickvals=x_labels[:: max(1, len(x_labels) // 12)],  # 12개 정도만 표시
                ticktext=[label for label in x_labels[:: max(1, len(x_labels) // 12)]],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    except Exception as e:
        print(f"Chart creation error: {str(e)}")
        return None


def create_comparison_chart(item, case_num, colors):
    """기존 함수는 3년치 차트로 대체"""
    return create_3years_chart(item, case_num, colors)


def create_all_cases_chart(filtered_results):
    """모든 케이스를 한 번에 보여주는 3년치 차트"""
    fig = make_subplots(
        rows=len(filtered_results),
        cols=1,
        subplot_titles=[f"케이스 {i+1}" for i in range(len(filtered_results))],
        vertical_spacing=0.08,
    )

    # 년도별 색상 정의
    year_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 월 매핑 (숫자 -> 한글)
    month_mapping = {
        "01": "1월",
        "02": "2월",
        "03": "3월",
        "04": "4월",
        "05": "5월",
        "06": "6월",
        "07": "7월",
        "08": "8월",
        "09": "9월",
        "10": "10월",
        "11": "11월",
        "12": "12월",
    }

    for i, item in enumerate(filtered_results):
        try:
            # 3년치 데이터와 표준값 가져오기
            years_data = item["input_data"]["3년치 데이터"]
            standard = item["ground_truth"]["standard"]

            # 문자열인 경우 파싱
            if isinstance(years_data, str):
                years_data = ast.literal_eval(years_data)

            # 년도 정렬
            sorted_years = sorted(years_data.keys())

            # 연속된 시간축과 데이터 생성
            x_labels = []
            actual_values = []
            standard_values = []

            for year in sorted_years:
                year_data = years_data[year]
                sorted_months = sorted(year_data.keys())

                for month in sorted_months:
                    x_labels.append(f"{year}-{month}")
                    actual_values.append(year_data[month])
                    # 표준값
                    month_korean = month_mapping.get(month, f"{int(month)}월")
                    standard_val = standard.get(month_korean, 0)
                    standard_values.append(standard_val)

            # 표준값 라인 추가
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=standard_values,
                    mode="lines+markers",
                    name="표준값" if i == 0 else None,
                    line=dict(color="#888888", width=2, dash="dot"),
                    marker=dict(size=4),
                    opacity=0.7,
                    showlegend=True if i == 0 else False,
                ),
                row=i + 1,
                col=1,
            )

            # 년도별 실제 데이터 추가
            for j, year in enumerate(sorted_years):
                year_data = years_data[year]
                sorted_months = sorted(year_data.keys())

                year_x_labels = [f"{year}-{month}" for month in sorted_months]
                year_values = [year_data[month] for month in sorted_months]

                color = year_colors[j % len(year_colors)]

                # 첫 번째 케이스에서만 범례 표시
                show_legend = i == 0
                legend_name = f"20{year}년" if show_legend else None

                fig.add_trace(
                    go.Scatter(
                        x=year_x_labels,
                        y=year_values,
                        mode="lines+markers",
                        name=legend_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        showlegend=show_legend,
                    ),
                    row=i + 1,
                    col=1,
                )

        except Exception as e:
            print(f"Error processing case {i}: {str(e)}")
            continue

    fig.update_layout(
        height=400 * len(filtered_results),
        title_text="전체 케이스 3년치 데이터 비교",
        showlegend=True,
    )

    # x축 설정 (각 서브플롯별로)
    for i in range(len(filtered_results)):
        fig.update_xaxes(tickangle=45, row=i + 1, col=1)

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
                        # _run.py와 동일한 방식으로 수정
                        years_data = outlier_item["input_data"]["3년치 데이터"]

                        # 안전한 데이터 변환
                        if isinstance(years_data, str):
                            try:
                                years_data = ast.literal_eval(years_data)
                            except (ValueError, SyntaxError):
                                print(
                                    f"Warning: Could not parse years_data: {years_data}"
                                )
                                progress = (index + 1) / total
                                progress_bar2.progress(progress)
                                status_text2.text(
                                    f"패턴 체크 중: {index + 1}/{total} ({progress:.1%})"
                                )
                                result_item = outlier_item.copy()
                                return result_item
                        elif not isinstance(years_data, dict):
                            print(
                                f"Warning: years_data is not a valid format: {type(years_data)}"
                            )
                            progress = (index + 1) / total
                            progress_bar2.progress(progress)
                            status_text2.text(
                                f"패턴 체크 중: {index + 1}/{total} ({progress:.1%})"
                            )
                            result_item = outlier_item.copy()
                            return result_item

                        pattern_result = await analyzer.pattern_checker(years_data)

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


def generate_html_report(filtered_results):
    """분석 결과를 HTML 리포트로 생성"""

    # 패턴 결과에서 'yes' 개수 계산하는 헬퍼 함수
    def count_pattern_yes(results):
        count = 0
        for item in results:
            pattern_result = item.get("pattern_result", {})
            result_value = getattr(pattern_result, "result", None)
            if result_value is None and isinstance(pattern_result, dict):
                result_value = pattern_result.get("result")
            if result_value == "yes":
                count += 1
        return count

    # HTML 템플릿 시작
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>이상치 분석 리포트</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .summary {{
                display: flex;
                justify-content: space-around;
                padding: 30px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            .summary-item {{
                text-align: center;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                min-width: 150px;
            }}
            .summary-item h3 {{
                margin: 0;
                font-size: 2.5em;
                color: #667eea;
                font-weight: bold;
            }}
            .summary-item p {{
                margin: 5px 0 0 0;
                color: #666;
                font-weight: 500;
            }}
            .content {{
                padding: 40px;
            }}
            .case-section {{
                margin-bottom: 50px;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                overflow: hidden;
            }}
            .case-header {{
                background-color: #667eea;
                color: white;
                padding: 20px;
                font-size: 1.3em;
                font-weight: bold;
            }}
            .case-content {{
                padding: 30px;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-top: 30px;
            }}
            .info-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .info-box h4 {{
                margin: 0 0 15px 0;
                color: #667eea;
                font-weight: bold;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .data-table th, .data-table td {{
                border: 1px solid #dee2e6;
                padding: 12px;
                text-align: left;
            }}
            .data-table th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            .data-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .judgment {{
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .judgment.yes {{
                background-color: #dc3545;
                color: white;
            }}
            .judgment.no {{
                background-color: #28a745;
                color: white;
            }}
            .footer {{
                text-align: center;
                padding: 30px;
                background-color: #f8f9fa;
                color: #666;
                border-top: 1px solid #dee2e6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 이상치 분석 리포트</h1>
                <p>생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-item">
                    <h3>{len(filtered_results)}</h3>
                    <p>최종 이상치</p>
                </div>
                <div class="summary-item">
                    <h3>{count_pattern_yes(filtered_results)}</h3>
                    <p>패턴 이상</p>
                </div>
                <div class="summary-item">
                    <h3>{len(set(item['ground_truth']['category'] for item in filtered_results))}</h3>
                    <p>카테고리 수</p>
                </div>
            </div>
            
            <div class="content">
    """

    # 색상 팔레트
    color_pairs = [
        ("#1f77b4", "#ff7f0e"),
        ("#2ca02c", "#d62728"),
        ("#9467bd", "#8c564b"),
        ("#e377c2", "#7f7f7f"),
        ("#bcbd22", "#17becf"),
        ("#aec7e8", "#ffbb78"),
    ]

    # 각 케이스별 섹션 생성
    for i, item in enumerate(filtered_results):
        pattern_result = item["pattern_result"]
        result_value = getattr(pattern_result, "result", None)
        reason_value = getattr(pattern_result, "reason", None)
        if result_value is None and isinstance(pattern_result, dict):
            result_value = pattern_result.get("result")
            reason_value = pattern_result.get("reason")

        # 차트 생성
        fig = create_comparison_chart(item, i + 1, color_pairs)
        chart_html = ""
        if fig:
            chart_html = fig.to_html(include_plotlyjs=False, div_id=f"chart_{i}")

        html_content += f"""
                <div class="case-section">
                    <div class="case-header">
                        케이스 {i+1} - {item['ground_truth']['category']} (구분: {item['input_data']['구분']})
                    </div>
                    <div class="case-content">
                        <div class="info-grid">
                            <div class="info-box">
                                <h4>📋 분석 정보</h4>
                                <p><strong>카테고리:</strong> {item['ground_truth']['category']}</p>
                                <p><strong>데이터 개수:</strong> {item['ground_truth']['data_num']}</p>
                                <p><strong>구분:</strong> {item['input_data']['구분']}</p>
                                <div class="judgment {'yes' if result_value == 'yes' else 'no'}">
                                    판정: {'이상' if result_value == 'yes' else '정상'}
                                </div>
                            </div>
                            <div class="info-box">
                                <h4>💡 분석 이유</h4>
                                <p>{reason_value if reason_value else '이유 정보 없음'}</p>
                            </div>
                        </div>
                        
                        <div class="chart-container">
                            <h4>📈 3년치 데이터 vs 표준값 비교 차트</h4>
                            {chart_html}
                        </div>
                        
                        <div class="info-grid">
                            <div class="info-box">
                                <h4>📊 기준 데이터</h4>
                                <table class="data-table">
                                    <tr><th>월</th><th>값</th></tr>
        """

        # 기준 데이터 테이블
        for month, value in item["standard_data"].items():
            html_content += f"<tr><td>{month}</td><td>{value:,.0f}</td></tr>"

        html_content += """
                                </table>
                            </div>
                            <div class="info-box">
                                <h4>📊 비교 데이터</h4>
                                <table class="data-table">
                                    <tr><th>월</th><th>값</th></tr>
        """

        # 비교 데이터 테이블
        for month, value in item["comparison_input_data"].items():
            html_content += f"<tr><td>{month}</td><td>{value:,.0f}</td></tr>"

        html_content += """
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
        """

    # HTML 마무리
    html_content += """
            </div>
            
            <div class="footer">
                <p>이상치 분석 시스템에서 생성된 리포트입니다.</p>
                <p>본 리포트는 AI 기반 분석 결과를 포함하고 있습니다.</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


def create_excel_report(filtered_results):
    """분석 결과를 Excel 리포트로 생성"""
    # 요약 데이터 준비
    summary_data = []
    detailed_data = []

    for i, item in enumerate(filtered_results):
        pattern_result = item["pattern_result"]
        result_value = getattr(pattern_result, "result", None)
        reason_value = getattr(pattern_result, "reason", None)
        if result_value is None and isinstance(pattern_result, dict):
            result_value = pattern_result.get("result")
            reason_value = pattern_result.get("reason")

        # 요약 데이터
        summary_data.append(
            {
                "케이스": i + 1,
                "카테고리": item["ground_truth"]["category"],
                "구분": item["input_data"]["구분"],
                "데이터 개수": item["ground_truth"]["data_num"],
                "판정 결과": "이상" if result_value == "yes" else "정상",
                "분석 이유": reason_value if reason_value else "이유 정보 없음",
            }
        )

        # 상세 데이터 (기준 vs 비교)
        standard_data = item["standard_data"]
        comparison_data = item["comparison_input_data"]

        for month in standard_data.keys():
            detailed_data.append(
                {
                    "케이스": i + 1,
                    "카테고리": item["ground_truth"]["category"],
                    "월": month,
                    "기준값": standard_data.get(month, 0),
                    "비교값": comparison_data.get(month, 0),
                    "차이": comparison_data.get(month, 0) - standard_data.get(month, 0),
                    "변화율(%)": (
                        (
                            (
                                comparison_data.get(month, 0)
                                - standard_data.get(month, 0)
                            )
                            / standard_data.get(month, 1)
                            * 100
                        )
                        if standard_data.get(month, 0) != 0
                        else 0
                    ),
                }
            )

    # DataFrame 생성
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)

    return summary_df, detailed_df


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

            # 기본 통계만 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("최종 이상치 케이스 수", len(filtered_results))
            with col2:
                pattern_yes_count = sum(
                    1
                    for item in filtered_results
                    if (
                        getattr(item.get("pattern_result", {}), "result", None) == "yes"
                        or (
                            isinstance(item.get("pattern_result", {}), dict)
                            and item.get("pattern_result", {}).get("result") == "yes"
                        )
                    )
                )
                st.metric("패턴 이상 케이스", pattern_yes_count)

            # 리포트 다운로드 섹션
            st.markdown("---")
            st.subheader("📥 분석 결과 다운로드")

            st.info(
                "🚀 **분석이 완료되었습니다!** 아래에서 상세한 결과를 다운로드하세요."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🌐 HTML 리포트**")
                st.write("📊 차트가 포함된 완전한 분석 리포트")
                st.write("✨ 이쁜 디자인으로 모든 결과를 한눈에 확인")

                # HTML 리포트 생성
                html_report = generate_html_report(filtered_results)

                st.download_button(
                    label="📄 HTML 리포트 다운로드",
                    data=html_report,
                    file_name=f"이상치분석리포트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    help="브라우저에서 열어서 차트와 함께 확인할 수 있습니다",
                    type="primary",
                )

            with col2:
                st.markdown("**📊 Excel 리포트**")
                st.write("📈 데이터 분석용 Excel 파일")
                st.write("🔍 추가 분석을 위한 상세 데이터")

                # Excel 리포트 생성
                summary_df, detailed_df = create_excel_report(filtered_results)

                # Excel 파일로 저장
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".xlsx"
                ) as tmp_file:
                    with pd.ExcelWriter(tmp_file.name, engine="openpyxl") as writer:
                        summary_df.to_excel(writer, sheet_name="요약", index=False)
                        detailed_df.to_excel(
                            writer, sheet_name="상세데이터", index=False
                        )

                    # 파일 읽어서 다운로드 버튼에 제공
                    with open(tmp_file.name, "rb") as f:
                        excel_data = f.read()

                    st.download_button(
                        label="📊 Excel 리포트 다운로드",
                        data=excel_data,
                        file_name=f"이상치분석데이터_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Excel에서 열어서 데이터를 추가 분석할 수 있습니다",
                    )

            # 다운로드 가이드
            st.markdown("---")
            st.success(
                """
            ✅ **분석 완료!** 위의 다운로드 버튼을 클릭하여 결과를 확인하세요.
            
            📋 **파일 설명:**
            - **HTML 리포트**: 브라우저에서 열어서 이쁜 차트와 함께 모든 결과를 한눈에 확인
            - **Excel 리포트**: '요약' 시트와 '상세데이터' 시트로 구성되어 추가 분석 가능
            """
            )

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

        # 3년치 샘플 데이터로 차트 생성
        sample_x_labels = [
            "22-04",
            "22-05",
            "22-06",
            "22-07",
            "22-08",
            "22-09",
            "23-01",
            "23-02",
            "23-03",
            "23-04",
            "23-05",
            "23-06",
            "24-01",
            "24-02",
            "24-03",
            "24-04",
            "24-05",
            "24-06",
        ]

        # 22년 데이터 (파란색)
        sample_22_data = [100, 110, 95, 120, 105, 115]
        sample_22_x = sample_x_labels[:6]

        # 23년 데이터 (주황색)
        sample_23_data = [98, 108, 92, 125, 102, 118]
        sample_23_x = sample_x_labels[6:12]

        # 24년 데이터 (초록색)
        sample_24_data = [85, 95, 80, 110, 88, 100]
        sample_24_x = sample_x_labels[12:18]

        # 표준값 (반복)
        sample_standard = [100, 105, 95, 115, 100, 110] * 3

        fig = go.Figure()

        # 표준값 라인 (회색, 점선)
        fig.add_trace(
            go.Scatter(
                x=sample_x_labels,
                y=sample_standard,
                mode="lines+markers",
                name="표준값 (반복)",
                line=dict(color="#888888", width=2, dash="dot"),
                marker=dict(size=6),
                opacity=0.7,
            )
        )

        # 22년 실제값
        fig.add_trace(
            go.Scatter(
                x=sample_22_x,
                y=sample_22_data,
                mode="lines+markers",
                name="2022년 실제값",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )

        # 23년 실제값
        fig.add_trace(
            go.Scatter(
                x=sample_23_x,
                y=sample_23_data,
                mode="lines+markers",
                name="2023년 실제값",
                line=dict(color="#ff7f0e", width=3),
                marker=dict(size=8),
            )
        )

        # 24년 실제값 (이상치 패턴)
        fig.add_trace(
            go.Scatter(
                x=sample_24_x,
                y=sample_24_data,
                mode="lines+markers",
                name="2024년 실제값 (이상치)",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="샘플 차트: 3년치 데이터 vs 표준값 비교",
            xaxis_title="시기 (년-월)",
            yaxis_title="값",
            template="plotly_white",
            height=400,
            xaxis=dict(tickangle=45),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
