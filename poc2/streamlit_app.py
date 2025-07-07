import streamlit as st
import pandas as pd
import json
import os
import asyncio
import sys
from io import StringIO
import tempfile
from tqdm import tqdm
import numpy as np
import ast

# 모델 경로 추가
MODEL_PATH = os.path.abspath("../")
sys.path.append(MODEL_PATH)

# 현재 디렉토리의 모듈들 임포트
from _run import Analyze, main as run_main
from utils import get_json, get_data_from_txt, get_heat_input_gt
from model import initialize_llm

# Streamlit 페이지 설정
st.set_page_config(
    page_title="가스 사용량 이상 데이터 분석 시스템",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바
st.sidebar.title("🔥 가스 분석 시스템")
st.sidebar.markdown("---")

# 메인 타이틀
st.title("가스 사용량 이상 데이터 분석 시스템")
st.markdown("업종별 가스 사용량 패턴을 분석하여 이상 데이터를 탐지합니다.")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3 = st.tabs(
    ["📊 기준 데이터 시각화", "🔍 이상 데이터 분석", "📈 분석 결과"]
)


# 캐시된 데이터 로딩 함수들
@st.cache_data
def load_ground_truth_data():
    """Ground truth 데이터 로딩"""
    ground_truth_path = os.path.join(
        os.path.dirname(__file__), "./make_instruction/group_biz_with_usage.json"
    )
    return get_json(ground_truth_path)


@st.cache_data
def load_heat_data():
    """Heat input 데이터 로딩"""
    group_heat_path = os.path.join(
        os.path.dirname(__file__), "./make_instruction/group_heat_input.xlsx"
    )
    return pd.read_excel(group_heat_path)


@st.cache_resource
def initialize_model():
    """LLM 모델 초기화"""
    return initialize_llm("langchain_gpt4o")


def convert_monthly_data_to_df(monthly_dict, title):
    """월별 딕셔너리 데이터를 DataFrame으로 변환"""
    if not monthly_dict:
        return pd.DataFrame()

    months = [
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

    # 딕셔너리의 키가 숫자(1-12)인 경우와 월 이름인 경우 모두 처리
    data = []
    for month in months:
        month_num = months.index(month) + 1
        value = monthly_dict.get(str(month_num)) or monthly_dict.get(month) or 0
        data.append(value)

    df = pd.DataFrame({"월": months, title: data})
    return df


def display_category_usage_data(ground_truth, selected_category):
    """선택된 카테고리의 사용량 데이터 표시"""
    if selected_category not in ground_truth:
        st.warning(f"'{selected_category}' 카테고리의 데이터가 없습니다.")
        return

    category_data = ground_truth[selected_category]

    # 용도별 데이터 표시
    for usage, usage_data in category_data.items():
        st.subheader(f"📍 {usage}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Median 값**")
            if "median" in usage_data:
                median_df = convert_monthly_data_to_df(usage_data["median"], "Median")
                if not median_df.empty:
                    st.dataframe(median_df, use_container_width=True)

                    # 차트 표시
                    st.line_chart(median_df.set_index("월")["Median"])

        with col2:
            st.write("**IQR 값**")
            if "iqr" in usage_data:
                iqr_df = convert_monthly_data_to_df(usage_data["iqr"], "IQR")
                if not iqr_df.empty:
                    st.dataframe(iqr_df, use_container_width=True)

                    # 차트 표시
                    st.line_chart(iqr_df.set_index("월")["IQR"])

        st.markdown("---")


def convert_string_to_dict(value):
    """문자열로 저장된 딕셔너리를 실제 딕셔너리로 변환"""
    try:
        if isinstance(value, str):
            # 먼저 ast.literal_eval로 시도
            return ast.literal_eval(value)
        elif isinstance(value, dict):
            return value
        else:
            return {}
    except:
        try:
            # JSON 파싱 시도 (작은따옴표를 큰따옴표로 변경)
            json_str = value.replace("'", '"')
            return json.loads(json_str)
        except:
            print(f"딕셔너리 파싱 실패: {value}")
            return {}


def save_results_to_txt(output_path, results):
    """분석 결과를 텍스트 파일로 저장"""
    outlier_results = [
        item for item in results if item["judge_result"].result == "이상"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"이상 데이터 분석 결과 ({len(outlier_results)}건)\n")
        f.write("=" * 50 + "\n\n")

        for i, item in enumerate(outlier_results, 1):
            f.write(f"[{i}번째 이상 사례]\n")
            f.write(f"결과: {item['judge_result'].result}\n")
            f.write(f"이유: {item['judge_result'].reason}\n")
            f.write(f"입력 데이터: {item['input_data']}\n")
            f.write("-" * 30 + "\n\n")

    return output_path


# Tab 1: 기준 데이터 시각화
with tab1:
    st.header("📊 기준 데이터 시각화")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏢 업종별 사용량 기준 데이터")

        # Ground truth 데이터 로딩
        try:
            ground_truth = load_ground_truth_data()

            # 카테고리 목록 추출 (그룹 정보에서)
            if "그룹" in ground_truth:
                categories = list(set(ground_truth["그룹"].values()))
                categories.sort()

                selected_category = st.selectbox(
                    "업종을 선택하세요:", categories, key="category_select"
                )

                if selected_category:
                    st.write(f"**선택된 업종: {selected_category}**")

                    # 선택된 카테고리에 해당하는 인덱스들 찾기
                    category_indices = []
                    for idx, group_name in ground_truth["그룹"].items():
                        if group_name == selected_category:
                            category_indices.append(idx)

                    if category_indices:
                        st.write(
                            f"**해당 업종의 데이터 항목: {len(category_indices)}개**"
                        )

                        # 용도별로 그룹화하여 표시
                        usage_groups = {}
                        for idx in category_indices:
                            if idx in ground_truth["용도"]:
                                usage = ground_truth["용도"][idx]
                                if usage not in usage_groups:
                                    usage_groups[usage] = []
                                usage_groups[usage].append(idx)

                        # 각 용도별 데이터 표시
                        for usage, indices in usage_groups.items():
                            st.subheader(f"📍 {usage}")

                            # 첫 번째 인덱스의 데이터를 표시 (대표값)
                            idx = indices[0]

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.write("**Median 값**")
                                if (
                                    "사용량 패턴 중앙값" in ground_truth
                                    and idx in ground_truth["사용량 패턴 중앙값"]
                                ):
                                    median_data = ground_truth["사용량 패턴 중앙값"][
                                        idx
                                    ]
                                    median_df = convert_monthly_data_to_df(
                                        median_data, "Median"
                                    )
                                    if not median_df.empty:
                                        st.dataframe(
                                            median_df, use_container_width=True
                                        )
                                        st.line_chart(
                                            median_df.set_index("월")["Median"]
                                        )
                                    else:
                                        st.write("데이터 없음")
                                else:
                                    st.write("Median 데이터 없음")

                            with col_b:
                                st.write("**IQR 값**")
                                if (
                                    "사용량 패턴 IQR" in ground_truth
                                    and idx in ground_truth["사용량 패턴 IQR"]
                                ):
                                    iqr_data = ground_truth["사용량 패턴 IQR"][idx]
                                    iqr_df = convert_monthly_data_to_df(iqr_data, "IQR")
                                    if not iqr_df.empty:
                                        st.dataframe(iqr_df, use_container_width=True)
                                        st.line_chart(iqr_df.set_index("월")["IQR"])
                                    else:
                                        st.write("데이터 없음")
                                else:
                                    st.write("IQR 데이터 없음")

                            st.markdown("---")
                    else:
                        st.warning(
                            f"'{selected_category}' 업종에 해당하는 데이터가 없습니다."
                        )

        except Exception as e:
            st.error(f"Ground truth 데이터 로딩 중 오류 발생: {str(e)}")

    with col2:
        st.subheader("🔥 열량별 사용량 기준 데이터")

        try:
            heat_data = load_heat_data()
            st.write("**열량 구간별 사용량 패턴**")

            # 데이터프레임 표시
            st.dataframe(heat_data, use_container_width=True)

            # 열량 구간 선택
            if "열량" in heat_data.columns:
                heat_ranges = heat_data["열량"].unique()
                selected_heat_range = st.selectbox(
                    "열량 구간을 선택하세요:", heat_ranges, key="heat_select"
                )

                if selected_heat_range:
                    selected_heat_data = heat_data[
                        heat_data["열량"] == selected_heat_range
                    ].iloc[0]

                    st.write(f"**선택된 열량 구간: {selected_heat_range}**")

                    # median과 IQR 데이터 표시
                    if "사용량_패턴_median" in selected_heat_data:
                        try:
                            median_data = convert_string_to_dict(
                                selected_heat_data["사용량_패턴_median"]
                            )
                            median_df = convert_monthly_data_to_df(
                                median_data, "Median"
                            )
                            if not median_df.empty:
                                st.write("**Median 사용량**")
                                st.dataframe(median_df, use_container_width=True)
                                st.line_chart(median_df.set_index("월")["Median"])
                        except Exception as e:
                            st.write("**Median 데이터 (파싱 실패)**")
                            st.write(
                                f"원본 데이터: {selected_heat_data['사용량_패턴_median']}"
                            )
                            st.write(f"오류: {str(e)}")

                    if "사용량_패턴_IQR" in selected_heat_data:
                        try:
                            iqr_data = convert_string_to_dict(
                                selected_heat_data["사용량_패턴_IQR"]
                            )
                            iqr_df = convert_monthly_data_to_df(iqr_data, "IQR")
                            if not iqr_df.empty:
                                st.write("**IQR 사용량**")
                                st.dataframe(iqr_df, use_container_width=True)
                                st.line_chart(iqr_df.set_index("월")["IQR"])
                        except Exception as e:
                            st.write("**IQR 데이터 (파싱 실패)**")
                            st.write(
                                f"원본 데이터: {selected_heat_data['사용량_패턴_IQR']}"
                            )
                            st.write(f"오류: {str(e)}")

        except Exception as e:
            st.error(f"Heat 데이터 로딩 중 오류 발생: {str(e)}")

# Tab 2: 이상 데이터 분석
with tab2:
    st.header("🔍 이상 데이터 분석")

    # 파일 업로드
    uploaded_file = st.file_uploader(
        "preprocessed.txt 파일을 업로드하세요",
        type=["txt"],
        help="분석할 가스 사용량 데이터 파일을 업로드하세요.",
    )

    if uploaded_file is not None:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp_file:
            content = uploaded_file.getvalue().decode("utf-8")
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        try:
            # 업로드된 파일에서 데이터 읽기
            data_lst = get_data_from_txt(temp_file_path)

            st.success(
                f"✅ 파일이 성공적으로 업로드되었습니다. (총 {len(data_lst)}개 데이터)"
            )

            # 데이터 미리보기
            if st.checkbox("데이터 미리보기", value=False):
                st.write("**업로드된 데이터 (처음 5개):**")
                preview_df = pd.DataFrame(data_lst[:5])
                st.dataframe(preview_df, use_container_width=True)

            # 분석 실행 버튼
            if st.button("🚀 이상 데이터 분석 시작", type="primary"):
                with st.spinner("분석 중입니다... 시간이 다소 소요될 수 있습니다."):

                    # 진행률 표시용 플레이스홀더
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # LLM 초기화
                        status_text.text("LLM 모델 초기화 중...")
                        llm = initialize_model()

                        # 분석기 초기화
                        analyzer = Analyze(llm)

                        # 비동기 분석 실행을 위한 함수
                        async def run_analysis():
                            results = []
                            total = len(data_lst)

                            # 세마포어로 동시 실행 수 제한
                            semaphore = asyncio.Semaphore(10)  # 동시 실행 수 줄임

                            async def process_with_progress(idx, data_item):
                                async with semaphore:
                                    result = await analyzer.process_single_item(
                                        data_item
                                    )
                                    progress = (idx + 1) / total
                                    progress_bar.progress(progress)
                                    status_text.text(
                                        f"진행률: {idx + 1}/{total} ({progress:.1%})"
                                    )
                                    return result

                            # 모든 작업을 비동기로 실행
                            tasks = [
                                process_with_progress(i, data_item)
                                for i, data_item in enumerate(data_lst)
                            ]
                            results = await asyncio.gather(*tasks)

                            return results

                        # 비동기 함수 실행
                        results = asyncio.run(run_analysis())

                        # 결과를 세션 상태에 저장
                        st.session_state["analysis_results"] = results
                        st.session_state["data_count"] = len(data_lst)

                        # 이상 데이터만 필터링
                        outlier_results = [
                            item
                            for item in results
                            if item["judge_result"].result == "이상"
                        ]
                        st.session_state["outlier_results"] = outlier_results

                        progress_bar.progress(1.0)
                        status_text.text("✅ 분석 완료!")

                        st.success(
                            f"🎉 분석이 완료되었습니다! 총 {len(results)}건 중 {len(outlier_results)}건의 이상 데이터를 발견했습니다."
                        )

                    except Exception as e:
                        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")

        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}")

        finally:
            # 임시 파일 삭제
            if "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

# Tab 3: 분석 결과
with tab3:
    st.header("📈 분석 결과")

    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        outlier_results = st.session_state["outlier_results"]
        data_count = st.session_state["data_count"]

        # 결과 요약
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("전체 데이터", f"{data_count:,}건")
        with col2:
            st.metric("이상 데이터", f"{len(outlier_results):,}건")
        with col3:
            outlier_rate = (
                (len(outlier_results) / data_count) * 100 if data_count > 0 else 0
            )
            st.metric("이상률", f"{outlier_rate:.1f}%")

        st.markdown("---")

        # 이상 데이터 상세 보기
        if outlier_results:
            st.subheader("🚨 이상 데이터 상세 결과")

            # 페이지네이션
            items_per_page = 5
            total_pages = (len(outlier_results) + items_per_page - 1) // items_per_page

            if total_pages > 1:
                page = st.selectbox(
                    "페이지 선택", range(1, total_pages + 1), key="page_select"
                )
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(outlier_results))
                current_results = outlier_results[start_idx:end_idx]
            else:
                current_results = outlier_results
                start_idx = 0

            # 이상 데이터 표시
            for i, item in enumerate(current_results):
                with st.expander(f"🔍 이상 사례 #{start_idx + i + 1}"):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write("**📋 기본 정보**")
                        input_data = item["input_data"]
                        st.write(f"**구분:** {input_data.get('구분', 'N/A')}")
                        st.write(f"**업태:** {input_data.get('업태', 'N/A')}")
                        st.write(f"**업종:** {input_data.get('업종', 'N/A')}")
                        st.write(f"**용도:** {input_data.get('용도', 'N/A')}")
                        st.write(
                            f"**보일러 열량:** {input_data.get('보일러 열량', 0):,}"
                        )
                        st.write(
                            f"**연소기 열량:** {input_data.get('연소기 열량', 0):,}"
                        )

                        st.write("**입력 데이터:**")
                        usage_pattern = convert_string_to_dict(
                            input_data.get("사용량_패턴", "{}")
                        )

                        # 월별 사용량 표시
                        months = [
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

                        # usage_df 생성
                        usage_data = []
                        for month in months:
                            usage = usage_pattern.get(month, 0)
                            usage_data.append({"월": month, "사용량": usage})
                            st.write(f"- {month}: {usage}")

                        usage_df = pd.DataFrame(usage_data)

                    with col2:
                        st.write("**⚠️ 이상 판단 결과**")
                        judge_result = item["judge_result"]
                        st.error(f"**판정:** {judge_result.result}")

                        st.write("**📝 이상 사유:**")
                        st.write(judge_result.reason)

                        # 사용량 차트
                        st.write("**📊 월별 사용량 그래프**")
                        if not usage_df.empty:
                            st.line_chart(usage_df.set_index("월")["사용량"])
                        else:
                            st.write("표시할 데이터가 없습니다.")

        # 결과 다운로드
        st.markdown("---")
        st.subheader("💾 결과 다운로드")

        # 다운로드할 내용 생성
        download_content = f"이상 데이터 분석 결과 ({len(outlier_results)}건)\n"
        download_content += "=" * 50 + "\n\n"

        for i, item in enumerate(outlier_results, 1):
            download_content += f"[{i}번째 이상 사례]\n"
            download_content += f"결과: {item['judge_result'].result}\n"
            download_content += f"이유: {item['judge_result'].reason}\n"
            download_content += f"입력 데이터: {item['input_data']}\n"
            download_content += "-" * 30 + "\n\n"

        # 다운로드 버튼
        st.download_button(
            label="📁 결과를 TXT 파일로 다운로드",
            data=download_content,
            file_name=f"outlier_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            type="primary",
        )

    else:
        st.info(
            "📝 분석을 먼저 실행해주세요. '이상 데이터 분석' 탭에서 파일을 업로드하고 분석을 시작하세요."
        )

# 사이드바 정보
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ 사용법")
st.sidebar.markdown(
    """
1. **기준 데이터 시각화**: 업종별, 열량별 기준 데이터를 확인
2. **이상 데이터 분석**: 파일 업로드 후 분석 실행
3. **분석 결과**: 결과 확인 및 다운로드
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 문의")
st.sidebar.markdown("분석 관련 문의사항이 있으시면 개발팀에 연락해주세요.")

# 스타일링
st.markdown(
    """
<style>
    .stMetric > div > div > div > div {
        color: #1f77b4;
    }
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
    }
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
    }
</style>
""",
    unsafe_allow_html=True,
)
