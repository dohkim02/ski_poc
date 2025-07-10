import streamlit as st
import pandas as pd
import time
import io
from run_data2 import analyze_all_rows_parallel


def main():
    st.set_page_config(
        page_title="계량기 이상 징후 분석", page_icon="📊", layout="wide"
    )

    st.title("📊 계량기 이상 징후 분석 (전체 데이터)")
    st.markdown("---")

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 분석 설정")

        # 병렬 처리 설정
        st.subheader("🚀 성능 설정")
        max_workers = st.slider(
            "병렬 처리 워커 수",
            min_value=1,
            max_value=16,
            value=4,
            help="동시에 처리할 프로세스 수 (높을수록 빠르지만 메모리 사용량 증가)",
        )

    # 파일 업로드
    uploaded_file = st.file_uploader(
        "📁 엑셀 파일 업로드",
        type=["xlsx"],
        help="분석할 계량기 데이터가 포함된 엑셀 파일을 업로드하세요",
    )

    if uploaded_file:
        # 설정 섹션
        st.subheader("🔧 분석 조건 설정")

        concept_options = [
            "계량기 당월 사용량",
            "계량기 전월 사용량",
            "계량기 전년동월 사용량",
        ]

        # 조건 설정
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_A = st.selectbox("조건 A", concept_options, key="A")

        with col2:
            filtered_B_options = [opt for opt in concept_options if opt != selected_A]
            selected_B = st.selectbox("조건 B", filtered_B_options, key="B")

        with col3:
            op = st.radio("A와 B 조건 연산자", ["and", "or"])

        # 임계값 설정
        st.subheader("📏 임계값 설정")
        col1, col2 = st.columns(2)

        with col1:
            threshold_value = st.number_input(
                "기준값 (A/B 사용량이 이 값 이상이어야 조건 만족)",
                value=100,
                min_value=0,
                help="선택한 조건 A와 B의 사용량이 이 값 이상이어야 분석 대상이 됩니다",
            )

        with col2:
            drop_ratio_percent = st.slider(
                "급감 판단 비율",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                format="%d%%",
                help="사용량이 이 비율 이하로 떨어지면 '급감'으로 판단합니다",
            )
            drop_ratio = drop_ratio_percent / 100.0  # 0-1 범위로 변환

        # 분석 실행 버튼
        if st.button("🚀 전체 데이터 분석 실행", type="primary"):

            # 진행률 표시를 위한 컨테이너들
            progress_container = st.container()
            status_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            with status_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    processed_metric = st.empty()
                with col2:
                    rate_metric = st.empty()
                with col3:
                    eta_metric = st.empty()

            try:
                start_time = time.time()

                # 진행률 콜백 함수
                def update_progress(progress, rate):
                    progress_bar.progress(progress)

                    processed_count = int(progress * 100)  # 임시로 100개 가정
                    status_text.text(f"처리 중... {progress:.1%} 완료")

                    processed_metric.metric("처리 진행률", f"{progress:.1%}")
                    rate_metric.metric("처리 속도", f"{rate:.1f} 행/초")

                    if rate > 0:
                        remaining_time = (
                            (1 - progress) / progress * (time.time() - start_time)
                        )
                        eta_metric.metric("예상 남은 시간", f"{remaining_time:.0f}초")

                # 병렬 분석 실행
                with st.spinner("데이터 분석 중..."):
                    result_df = analyze_all_rows_parallel(
                        uploaded_file,
                        selected_A,
                        selected_B,
                        op,
                        threshold_value,
                        drop_ratio,
                        max_workers=max_workers,
                        progress_callback=update_progress,
                    )

                # 처리 완료
                total_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text("✅ 분석 완료!")

                # 결과를 세션 상태에 저장
                st.session_state.result_df = result_df
                st.session_state.analysis_settings = {
                    "selected_A": selected_A,
                    "selected_B": selected_B,
                    "operator": op,
                    "threshold": threshold_value,
                    "drop_ratio": drop_ratio,
                    "processing_time": total_time,
                }

            except Exception as e:
                st.error(f"❌ 분석 중 오류 발생: {str(e)}")
                st.exception(e)

        # 결과 표시 (세션 상태에 결과가 있는 경우)
        if (
            hasattr(st.session_state, "result_df")
            and st.session_state.result_df is not None
        ):
            display_results()

    else:
        # 사용 방법 안내
        st.info("📋 사용 방법")
        st.markdown(
            """
        1. **파일 업로드**: 계량기 데이터가 포함된 엑셀 파일을 업로드하세요
        2. **조건 설정**: 분석할 조건 A, B와 연산자를 선택하세요
        3. **임계값 설정**: 기준값과 급감 판단 비율을 설정하세요
        4. **분석 실행**: "전체 데이터 분석 실행" 버튼을 클릭하세요
        5. **결과 확인**: 분석 결과를 확인하고 다운로드하세요
        
        **📊 분석 결과 유형:**
        - ✅ **정상**: 조건을 만족하고 급감하지 않은 경우
        - ⚠️ **전월 대비 급감**: 전월 대비 사용량이 급감한 경우
        - ⚠️ **전년동월 대비 급감**: 전년 동월 대비 사용량이 급감한 경우  
        - ⚠️ **평균 대비 급감**: 평균 사용량 대비 급감한 경우
        - 🚫 **조건 불충족**: 설정한 조건 A, B를 만족하지 않는 경우
        - 📵 **미사용세대**: 당월 사용량이 0인 경우
        """
        )


def display_results():
    """분석 결과를 표시하는 함수"""
    result_df = st.session_state.result_df
    settings = st.session_state.analysis_settings

    st.markdown("---")
    st.subheader("📊 분석 결과")

    # 처리 시간 및 설정 정보
    st.info(
        f"""
    **분석 완료!** 
    - 처리 시간: {settings['processing_time']:.2f}초
    - 분석 조건: {settings['selected_A']} {settings['operator']} {settings['selected_B']} ≥ {settings['threshold']}
    - 급감 기준: {int(settings['drop_ratio'] * 100)}% 이하
    - 총 처리 행 수: {len(result_df):,}개
    """
    )

    # 결과 통계
    st.subheader("📈 결과 통계")

    # 이상징후 결과별 집계
    result_counts = result_df["이상징후_결과"].value_counts()

    # 메트릭 표시
    total_rows = len(result_df)
    normal_count = result_counts.get("정상", 0)
    abnormal_count = (
        total_rows
        - normal_count
        - result_counts.get("조건 불충족", 0)
        - result_counts.get("데이터 없음", 0)
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("총 행 수", f"{total_rows:,}")
    with col2:
        st.metric(
            "정상", f"{normal_count:,}", delta=f"{normal_count/total_rows*100:.1f}%"
        )
    with col3:
        st.metric(
            "이상 징후",
            f"{abnormal_count:,}",
            delta=f"{abnormal_count/total_rows*100:.1f}%",
        )
    with col4:
        st.metric("조건 불충족", f"{result_counts.get('조건 불충족', 0):,}")
    with col5:
        st.metric("미사용세대", f"{result_counts.get('미사용세대', 0):,}")

    # 상세 결과 분포
    st.subheader("🔍 상세 결과 분포")

    # 결과별 통계 테이블
    result_stats = []
    for result_type, count in result_counts.items():
        percentage = (count / total_rows) * 100
        result_stats.append(
            {
                "결과 유형": result_type,
                "개수": f"{count:,}",
                "비율": f"{percentage:.2f}%",
            }
        )

    result_stats_df = pd.DataFrame(result_stats)
    st.dataframe(result_stats_df, use_container_width=True)

    # 데이터 필터링 옵션
    st.subheader("📋 결과 데이터")

    col1, col2 = st.columns(2)
    with col1:
        show_filter = st.selectbox(
            "표시할 데이터 선택",
            [
                "전체",
                "정상",
                "이상 징후만",
                "전월 대비 급감",
                "전년동월 대비 급감",
                "평균 대비 급감",
                "조건 불충족",
                "미사용세대",
            ],
        )

    with col2:
        show_details = st.checkbox("상세 분석 컬럼 표시", value=False)

    # 필터링 적용
    if show_filter == "전체":
        display_df = result_df
    elif show_filter == "정상":
        display_df = result_df[result_df["이상징후_결과"] == "정상"]
    elif show_filter == "이상 징후만":
        abnormal_conditions = [
            "전월 대비 급감",
            "전년동월 대비 급감",
            "사용량 평균 대비 급감",
        ]
        display_df = result_df[result_df["이상징후_결과"].isin(abnormal_conditions)]
    else:
        display_df = result_df[result_df["이상징후_결과"] == show_filter]

    # 표시할 컬럼 선택
    if show_details:
        # 모든 컬럼 표시
        st.dataframe(display_df, use_container_width=True)
    else:
        # 주요 컬럼만 표시
        essential_columns = [
            "이상징후_결과",
            "당월사용량",
            "전월사용량",
            "전년동월사용량",
            "평균사용량",
            "조건만족여부",
        ]
        available_columns = [
            col for col in essential_columns if col in display_df.columns
        ]
        other_columns = [
            col for col in display_df.columns if col not in essential_columns
        ][
            :5
        ]  # 처음 5개 원본 컬럼
        display_columns = other_columns + available_columns
        st.dataframe(display_df[display_columns], use_container_width=True)

    st.write(f"현재 표시 중: {len(display_df):,}개 행")

    # 다운로드 섹션
    st.subheader("💾 결과 다운로드")

    col1, col2 = st.columns(2)

    with col1:
        # 전체 결과 다운로드
        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📄 전체 결과 CSV 다운로드",
            data=csv,
            file_name=f"계량기_이상징후_분석결과_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with col2:
        # 이상 징후만 다운로드
        abnormal_only = result_df[
            ~result_df["이상징후_결과"].isin(["정상", "조건 불충족", "데이터 없음"])
        ]
        if len(abnormal_only) > 0:
            csv_abnormal = abnormal_only.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⚠️ 이상 징후만 CSV 다운로드",
                data=csv_abnormal,
                file_name=f"계량기_이상징후만_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.button("⚠️ 이상 징후 없음", disabled=True)

    # 새로 분석하기 버튼
    if st.button("🔄 새로운 파일로 다시 분석"):
        # 세션 상태 초기화
        if "result_df" in st.session_state:
            del st.session_state.result_df
        if "analysis_settings" in st.session_state:
            del st.session_state.analysis_settings
        st.rerun()


if __name__ == "__main__":
    main()
