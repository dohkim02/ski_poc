import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import time
import io
from datetime import datetime


def process_row(row_data, uncorrected_threshold=30, factor_threshold=0.04):
    """각 행을 처리하는 함수"""
    try:
        # 컬럼 이름을 키로 하는 딕셔너리로 변환
        row = row_data
        flags = []

        # 1. 비보정지침 이상 체크
        try:
            meter_current = row.get("계량기-당월지침", 0)
            uncorrected_current = row.get("비보정-당월지침", 0)
            if pd.notna(meter_current) and pd.notna(uncorrected_current):
                if meter_current - uncorrected_current >= uncorrected_threshold:
                    flags.append("비보정지침 이상")
        except:
            pass

        # 2. 전년동월팩터 이상 체크
        try:
            last_year_factor = row.get("전년동월팩터", 0)
            current_factor = row.get("당월팩터", 0)
            if (
                pd.notna(last_year_factor)
                and pd.notna(current_factor)
                and current_factor != 0
            ):
                ratio = abs(last_year_factor - current_factor) / current_factor
                if ratio >= factor_threshold:
                    flags.append("전년동월팩터 이상")
        except:
            pass

        # 3. 전월팩터 이상 체크
        try:
            last_month_factor = row.get("전월팩터", 0)
            current_factor = row.get("당월팩터", 0)
            if (
                pd.notna(last_month_factor)
                and pd.notna(current_factor)
                and current_factor != 0
            ):
                ratio = abs(last_month_factor - current_factor) / current_factor
                if ratio >= factor_threshold:
                    flags.append("전월팩터 이상")
        except:
            pass

        # 4. 전3개월평균팩터 이상 체크
        try:
            three_month_avg_factor = row.get("전3개월평균팩터", 0)
            current_factor = row.get("당월팩터", 0)
            if (
                pd.notna(three_month_avg_factor)
                and pd.notna(current_factor)
                and current_factor != 0
            ):
                ratio = abs(three_month_avg_factor - current_factor) / current_factor
                if ratio >= factor_threshold:
                    flags.append("3개월 평균 팩터 이상")
        except:
            pass

        # 5. 측정팩터 이상 체크
        try:
            measure_factor = row.get("측정팩터", 0)
            current_factor = row.get("당월팩터", 0)
            if (
                pd.notna(measure_factor)
                and pd.notna(current_factor)
                and current_factor != 0
            ):
                ratio = abs(measure_factor - current_factor) / current_factor
                if ratio >= factor_threshold:
                    flags.append("측정팩터 이상")
        except:
            pass

        # 플래그들을 문자열로 결합
        return ", ".join(flags) if flags else ""

    except Exception as e:
        return ""


def process_excel_parallel(
    df,
    uncorrected_threshold=30,
    factor_threshold=0.04,
    max_workers=4,
    progress_callback=None,
):
    """
    데이터프레임을 병렬로 처리하는 메인 함수
    """

    # 필요한 컬럼들이 있는지 확인
    required_columns = [
        "계량기-당월지침",
        "비보정-당월지침",
        "전년동월팩터",
        "당월팩터",
        "전월팩터",
        "전3개월평균팩터",
        "측정팩터",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"다음 컬럼들이 없습니다: {missing_columns}")
        st.info(f"사용 가능한 컬럼들: {list(df.columns)}")

    # 데이터를 딕셔너리 형태로 변환
    row_data_list = df.to_dict("records")

    # 병렬 처리
    total_rows = len(row_data_list)
    results = []
    completed_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 모든 행을 병렬로 처리
        future_to_index = {
            executor.submit(
                process_row, row, uncorrected_threshold, factor_threshold
            ): i
            for i, row in enumerate(row_data_list)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
                completed_count += 1

                # 진행률 업데이트
                if progress_callback:
                    progress = completed_count / total_rows
                    elapsed_time = time.time() - start_time
                    rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                    progress_callback(progress, rate)

            except Exception as e:
                results.append((index, ""))
                completed_count += 1

    # 결과를 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])
    flags = [result[1] for result in results]

    # 새로운 컬럼 추가
    df["이상치_플래그"] = flags

    return df


def main():
    st.set_page_config(page_title="보정비율 분석 시스템", page_icon="🔍", layout="wide")

    st.title("🔍 보정비율 분석 시스템")
    st.markdown("---")

    # 사이드바 - 설정
    st.sidebar.header("⚙️ 설정")

    # 임계값 설정
    st.sidebar.subheader("임계값 설정")
    uncorrected_threshold = st.sidebar.number_input(
        "비보정지침 임계값",
        min_value=0,
        max_value=1000,
        value=30,
        help="계량기-당월지침과 비보정-당월지침의 차이 임계값",
    )

    factor_threshold = (
        st.sidebar.number_input(
            "팩터 차이 임계값 (%)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.1,
            help="팩터 간의 차이 비율 임계값",
        )
        / 100
    )  # 백분율을 소수로 변환

    # 병렬 처리 설정
    st.sidebar.subheader("성능 설정")
    max_workers = st.sidebar.slider(
        "병렬 처리 워커 수",
        min_value=1,
        max_value=os.cpu_count() or 4,
        value=4,
        help="병렬 처리에 사용할 워커 수",
    )

    # 메인 영역
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 파일 업로드")
        uploaded_file = st.file_uploader(
            "Excel 파일을 선택하세요",
            type=["xlsx", "xls"],
            help="Excel 파일 (.xlsx 또는 .xls)을 업로드하세요",
        )

    with col2:
        st.header("📊 현재 설정")
        st.info(
            f"""
        **임계값 설정:**
        - 비보정지침: {uncorrected_threshold}
        - 팩터 차이: {factor_threshold*100:.1f}%
        
        **성능 설정:**
        - 워커 수: {max_workers}
        """
        )

    if uploaded_file is not None:
        try:
            # 파일 정보 표시
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.success(
                f"✅ 파일 업로드 완료: {uploaded_file.name} ({file_size:.2f} MB)"
            )

            # 데이터 로드
            with st.spinner("📖 데이터 로딩 중..."):
                df = pd.read_excel(uploaded_file)

            st.success(f"📊 데이터 로드 완료: {len(df):,}행, {len(df.columns)}컬럼")

            # 데이터 미리보기
            st.subheader("📋 데이터 미리보기")
            st.dataframe(df.head(), use_container_width=True)

            # 처리 시작 버튼
            if st.button("🚀 이상치 탐지 시작", type="primary"):
                start_time = time.time()

                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, rate):
                    progress_bar.progress(progress)
                    status_text.text(
                        f"처리 중... {progress*100:.1f}% 완료 (속도: {rate:.1f}행/초)"
                    )

                # 처리 실행
                with st.spinner("🔄 이상치 탐지 진행 중..."):
                    processed_df = process_excel_parallel(
                        df.copy(),
                        uncorrected_threshold=uncorrected_threshold,
                        factor_threshold=factor_threshold,
                        max_workers=max_workers,
                        progress_callback=update_progress,
                    )

                # 처리 완료
                total_time = time.time() - start_time
                st.success(f"✅ 처리 완료! (소요 시간: {total_time:.2f}초)")

                # 결과를 session_state에 저장
                st.session_state.processed_df = processed_df
                st.session_state.processing_time = total_time
                st.session_state.is_processed = True

            # 처리 결과가 있는 경우 결과 표시
            if (
                hasattr(st.session_state, "is_processed")
                and st.session_state.is_processed
            ):
                processed_df = st.session_state.processed_df

                # 결과 통계
                st.subheader("📊 이상치 탐지 결과")

                col1, col2, col3, col4 = st.columns(4)

                total_rows = len(processed_df)
                abnormal_rows = len(processed_df[processed_df["이상치_플래그"] != ""])
                normal_rows = total_rows - abnormal_rows
                abnormal_rate = (
                    (abnormal_rows / total_rows) * 100 if total_rows > 0 else 0
                )

                with col1:
                    st.metric("총 행 수", f"{total_rows:,}")

                with col2:
                    st.metric("이상치 발견", f"{abnormal_rows:,}")

                with col3:
                    st.metric("정상 행 수", f"{normal_rows:,}")

                with col4:
                    st.metric("이상치 비율", f"{abnormal_rate:.2f}%")

                # 이상치 유형별 통계
                if abnormal_rows > 0:
                    st.subheader("🔍 이상치 유형별 분석")

                    flag_counts = processed_df["이상치_플래그"].value_counts()
                    flag_data = []

                    for flag, count in flag_counts.items():
                        if flag != "":
                            percentage = (count / total_rows) * 100
                            flag_data.append(
                                {
                                    "이상치 유형": flag,
                                    "개수": count,
                                    "비율(%)": f"{percentage:.2f}%",
                                }
                            )

                    if flag_data:
                        flag_df = pd.DataFrame(flag_data)
                        st.dataframe(flag_df, use_container_width=True)

                # 결과 데이터 미리보기
                st.subheader("📋 처리 결과 미리보기")

                # 이상치만 보기 옵션
                show_only_abnormal = st.checkbox(
                    "이상치만 보기", key="show_abnormal_filter"
                )

                if show_only_abnormal:
                    display_df = processed_df[processed_df["이상치_플래그"] != ""]
                    st.write(f"이상치 {len(display_df):,}개 표시")
                else:
                    display_df = processed_df
                    st.write(f"전체 {len(display_df):,}개 표시")

                st.dataframe(display_df, use_container_width=True)

                # 다운로드 버튼
                st.subheader("💾 결과 다운로드")

                col1, col2 = st.columns(2)

                with col1:
                    # 전체 결과 다운로드
                    output_buffer = io.BytesIO()
                    processed_df.to_excel(output_buffer, index=False)
                    output_buffer.seek(0)

                    # 파일명 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"이상치_탐지_결과_{timestamp}.xlsx"

                    st.download_button(
                        label="📥 전체 결과 다운로드 (Excel)",
                        data=output_buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="원본 데이터 + 이상치_플래그 컬럼이 추가된 전체 결과",
                    )

                with col2:
                    # 이상치만 다운로드
                    if abnormal_rows > 0:
                        abnormal_buffer = io.BytesIO()
                        abnormal_df = processed_df[processed_df["이상치_플래그"] != ""]
                        abnormal_df.to_excel(abnormal_buffer, index=False)
                        abnormal_buffer.seek(0)

                        abnormal_filename = f"이상치만_{timestamp}.xlsx"

                        st.download_button(
                            label="🚨 이상치만 다운로드 (Excel)",
                            data=abnormal_buffer,
                            file_name=abnormal_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="이상치가 발견된 행만 포함된 결과",
                        )
                    else:
                        st.info("이상치가 없어 다운로드할 내용이 없습니다.")

                st.info(
                    f"""
                **다운로드 파일 정보:**
                - 원본 데이터의 모든 컬럼 + **이상치_플래그** 컬럼 추가
                - 이상치_플래그: 발견된 이상치 유형들이 쉼표로 구분되어 표시
                - 정상 데이터는 이상치_플래그가 빈 문자열("")로 표시
                """
                )

                # 새로 처리하기 버튼
                if st.button("🔄 새로운 파일로 다시 처리하기"):
                    # session_state 초기화
                    if "processed_df" in st.session_state:
                        del st.session_state.processed_df
                    if "processing_time" in st.session_state:
                        del st.session_state.processing_time
                    if "is_processed" in st.session_state:
                        del st.session_state.is_processed
                    st.rerun()

        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.exception(e)

    else:
        st.info("📁 Excel 파일을 업로드하여 이상치 탐지를 시작하세요.")

        # 사용 방법 안내
        st.subheader("📖 사용 방법")
        st.markdown(
            """
        1. **파일 업로드**: 좌측에서 Excel 파일을 업로드하세요
        2. **설정 조정**: 사이드바에서 임계값과 성능 설정을 조정하세요
        3. **탐지 시작**: "이상치 탐지 시작" 버튼을 클릭하세요
        4. **결과 확인**: 탐지 결과를 확인하고 필요시 다운로드하세요
        
        **필요한 컬럼:**
        - 계량기-당월지침
        - 비보정-당월지침
        - 전년동월팩터
        - 당월팩터
        - 전월팩터
        - 전3개월평균팩터
        - 측정팩터
        """
        )


if __name__ == "__main__":
    main()
