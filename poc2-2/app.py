import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import time
import io


def process_row(row_data, thresholds):
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
                if (
                    meter_current - uncorrected_current
                    >= thresholds["uncorrected_diff"]
                ):
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
                if (
                    abs(last_year_factor - current_factor) / current_factor
                    >= thresholds["factor_change_rate"]
                ):
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
                if (
                    abs(last_month_factor - current_factor) / current_factor
                    >= thresholds["factor_change_rate"]
                ):
                    flags.append("전월팩터 이상")
        except:
            pass

        # 4. 미사용세대 체크
        try:
            last_month_usage = row.get("계량기-전월사용량", 0)
            last_year_usage = row.get("계량기-전년동월사용량", 0)
            current_usage = row.get("계량기-당월사용량", 0)

            if (
                pd.notna(last_month_usage)
                and pd.notna(last_year_usage)
                and pd.notna(current_usage)
                and last_month_usage >= thresholds["min_usage_threshold"]
                and last_year_usage >= thresholds["min_usage_threshold"]
                and current_usage == thresholds["zero_usage"]
            ):
                flags.append("미사용세대")
        except:
            pass

        # 5. 전월 대비 급감 체크
        try:
            last_month_usage = row.get("계량기-전월사용량", 0)
            last_year_usage = row.get("계량기-전년동월사용량", 0)
            current_usage = row.get("계량기-당월사용량", 0)

            if (
                pd.notna(last_month_usage)
                and pd.notna(last_year_usage)
                and pd.notna(current_usage)
                and last_month_usage >= thresholds["min_usage_threshold"]
                and last_year_usage >= thresholds["min_usage_threshold"]
                and last_month_usage != 0
            ):
                ratio = current_usage / last_month_usage
                if ratio <= thresholds["usage_decrease_rate"]:
                    flags.append("전월 대비 급감")
        except:
            pass

        # 6. 전년동월 대비 급감 체크
        try:
            last_month_usage = row.get("계량기-전월사용량", 0)
            last_year_usage = row.get("계량기-전년동월사용량", 0)
            current_usage = row.get("계량기-당월사용량", 0)

            if (
                pd.notna(last_month_usage)
                and pd.notna(last_year_usage)
                and pd.notna(current_usage)
                and last_month_usage >= thresholds["min_usage_threshold"]
                and last_year_usage >= thresholds["min_usage_threshold"]
                and last_year_usage != 0
            ):
                ratio = current_usage / last_year_usage
                if ratio <= thresholds["usage_decrease_rate"]:
                    flags.append("전년동월 대비 급감")
        except:
            pass

        # 플래그들을 문자열로 결합
        return ", ".join(flags) if flags else ""

    except Exception as e:
        return ""


def process_excel_streamlit(df, max_workers=None, thresholds=None):
    """
    Streamlit용 엑셀 데이터 처리 함수
    """
    # 필요한 컬럼들이 있는지 확인
    required_columns = [
        "계량기-당월지침",
        "비보정-당월지침",
        "전년동월팩터",
        "당월팩터",
        "전월팩터",
        "계량기-전월사용량",
        "계량기-전년동월사용량",
        "계량기-당월사용량",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"⚠️ 다음 컬럼들이 없습니다: {missing_columns}")
        st.info(f"📋 사용 가능한 컬럼들: {list(df.columns)}")

    # 데이터를 딕셔너리 형태로 변환
    row_data_list = df.to_dict("records")

    # 병렬 처리
    total_rows = len(row_data_list)
    max_workers = max_workers or min(4, os.cpu_count())

    st.info(f"🚀 병렬 처리 시작 (워커 수: {max_workers}, 총 {total_rows:,}행)")

    # 진행률 표시를 위한 progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    completed_count = 0
    start_time = time.time()

    # 부분 함수로 thresholds를 바인딩
    process_row_with_thresholds = partial(process_row, thresholds=thresholds)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 모든 행을 병렬로 처리
        future_to_index = {
            executor.submit(process_row_with_thresholds, row): i
            for i, row in enumerate(row_data_list)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
                completed_count += 1

                # 진행률 업데이트
                progress = completed_count / total_rows
                progress_bar.progress(progress)

                # 현재 시간과 속도 계산
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    rate = completed_count / elapsed_time
                    status_text.text(
                        f"처리 중... {completed_count}/{total_rows} ({rate:.1f}행/초)"
                    )

            except Exception as e:
                results.append((index, ""))
                completed_count += 1
                progress = completed_count / total_rows
                progress_bar.progress(progress)

    # 결과를 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])
    flags = [result[1] for result in results]

    # 새로운 컬럼 추가
    df_result = df.copy()
    df_result["이상치_플래그"] = flags

    # 처리 완료 메시지
    total_time = time.time() - start_time
    status_text.text(f"✅ 처리 완료! 총 처리 시간: {total_time:.2f}초")
    progress_bar.progress(1.0)

    return df_result, total_time


def main():
    st.set_page_config(
        page_title="엑셀 이상치 탐지 도구", page_icon="📊", layout="wide"
    )

    st.title("📊 엑셀 이상치 탐지 도구")
    st.markdown("---")

    # 사이드바에 설정 옵션
    st.sidebar.header("⚙️ 설정")

    # 병렬 처리 워커 수
    max_workers = st.sidebar.slider("병렬 처리 워커 수", 1, os.cpu_count(), 4)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 임계값 설정")

    # 임계값 설정
    with st.sidebar.expander("임계값 설정", expanded=True):
        uncorrected_diff = st.number_input(
            "비보정지침 차이 임계값",
            min_value=0,
            max_value=1000,
            value=30,
            help="계량기-당월지침과 비보정-당월지침의 차이가 이 값 이상이면 이상치로 탐지",
        )

        factor_change_rate = (
            st.number_input(
                "팩터 변화율 임계값 (%)",
                min_value=0.0,
                max_value=100.0,
                value=4.0,
                step=0.1,
                help="팩터의 변화율이 이 비율 이상이면 이상치로 탐지",
            )
            / 100
        )  # 퍼센트를 소수로 변환

        min_usage_threshold = st.number_input(
            "최소 사용량 임계값",
            min_value=0,
            max_value=1000,
            value=100,
            help="이 값 이상의 사용량이 있어야 급감 체크 대상이 됨",
        )

        usage_decrease_rate = (
            st.number_input(
                "사용량 급감 임계값 (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="사용량이 이 비율 이하로 감소하면 급감으로 탐지",
            )
            / 100
        )  # 퍼센트를 소수로 변환

        zero_usage = st.number_input(
            "미사용 기준값",
            min_value=0,
            max_value=10,
            value=0,
            help="이 값과 같으면 미사용으로 판단",
        )

    # 임계값 딕셔너리 생성
    thresholds = {
        "uncorrected_diff": uncorrected_diff,
        "factor_change_rate": factor_change_rate,
        "min_usage_threshold": min_usage_threshold,
        "usage_decrease_rate": usage_decrease_rate,
        "zero_usage": zero_usage,
    }

    # 현재 설정값 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 현재 설정값")
    st.sidebar.markdown(f"• 비보정지침 차이: **{uncorrected_diff}**")
    st.sidebar.markdown(f"• 팩터 변화율: **{factor_change_rate*100:.1f}%**")
    st.sidebar.markdown(f"• 최소 사용량: **{min_usage_threshold}**")
    st.sidebar.markdown(f"• 급감 기준: **{usage_decrease_rate*100:.0f}%** 이하")
    st.sidebar.markdown(f"• 미사용 기준: **{zero_usage}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 탐지 항목")
    st.sidebar.markdown(
        """
    - 비보정지침 이상
    - 전년동월팩터 이상
    - 전월팩터 이상
    - 미사용세대
    - 전월 대비 급감
    - 전년동월 대비 급감
    """
    )

    # 파일 업로드
    st.header("📁 파일 업로드")
    uploaded_file = st.file_uploader(
        "엑셀 파일을 업로드하세요", type=["xlsx", "xls"], help="지원 형식: .xlsx, .xls"
    )

    if uploaded_file is not None:
        try:
            # 파일 정보 표시
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.success(f"✅ 파일 업로드 완료! 크기: {file_size:.2f} MB")

            # 엑셀 파일 읽기
            with st.spinner("📖 엑셀 파일을 읽는 중..."):
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ 데이터 로드 완료: {len(df):,}행, {len(df.columns)}컬럼")

            # 데이터 미리보기
            st.header("👀 데이터 미리보기")
            with st.expander("처음 5행 보기", expanded=True):
                st.dataframe(df.head(), use_container_width=True)

            # 처리 실행 버튼
            st.header("🚀 이상치 탐지 실행")
            if st.button("이상치 탐지 시작", type="primary"):
                with st.spinner("🔍 이상치 탐지를 실행 중입니다..."):
                    df_result, processing_time = process_excel_streamlit(
                        df, max_workers, thresholds
                    )

                # 결과 표시
                st.header("📊 처리 결과")

                # 통계 정보
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📋 총 행 수", f"{len(df_result):,}")

                with col2:
                    abnormal_count = len(df_result[df_result["이상치_플래그"] != ""])
                    st.metric("🚨 이상치 발견", f"{abnormal_count:,}")

                with col3:
                    normal_count = len(df_result[df_result["이상치_플래그"] == ""])
                    st.metric("✅ 정상 데이터", f"{normal_count:,}")

                with col4:
                    abnormal_rate = (abnormal_count / len(df_result)) * 100
                    st.metric("📊 이상치 비율", f"{abnormal_rate:.2f}%")

                # 이상치 유형별 통계
                st.subheader("🔍 이상치 유형별 분포")
                flag_counts = df_result["이상치_플래그"].value_counts()
                if len(flag_counts) > 1:  # 빈 문자열 제외하고 이상치가 있는 경우
                    flag_stats = []
                    for flag, count in flag_counts.items():
                        if flag != "":
                            percentage = (count / len(df_result)) * 100
                            flag_stats.append(
                                {
                                    "이상치 유형": flag,
                                    "개수": count,
                                    "비율(%)": f"{percentage:.2f}%",
                                }
                            )

                    if flag_stats:
                        st.dataframe(pd.DataFrame(flag_stats), use_container_width=True)
                else:
                    st.info("🎉 이상치가 발견되지 않았습니다!")

                # 처리 시간 정보
                st.subheader("⏱️ 처리 성능")
                avg_speed = len(df_result) / processing_time
                st.info(
                    f"총 처리 시간: {processing_time:.2f}초 | 평균 처리 속도: {avg_speed:.1f}행/초"
                )

                # 결과 데이터 미리보기
                st.subheader("📋 결과 데이터")
                with st.expander("결과 데이터 미리보기 (처음 10행)", expanded=True):
                    st.dataframe(df_result.head(10), use_container_width=True)

                # 이상치만 필터링해서 보기
                if abnormal_count > 0:
                    with st.expander(
                        f"🚨 이상치 데이터만 보기 ({abnormal_count}개)", expanded=False
                    ):
                        abnormal_data = df_result[df_result["이상치_플래그"] != ""]
                        st.dataframe(abnormal_data, use_container_width=True)

                # 파일 다운로드
                st.header("💾 결과 파일 다운로드")

                # 엑셀 파일로 변환
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_result.to_excel(writer, index=False, sheet_name="이상치탐지결과")

                excel_data = output.getvalue()

                # 다운로드 버튼
                st.download_button(
                    label="📥 처리된 엑셀 파일 다운로드",
                    data=excel_data,
                    file_name=f"이상치탐지결과_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )

                st.success("🎉 모든 작업이 완료되었습니다!")

        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.info("💡 파일 형식이 올바른지 확인해주세요.")

    else:
        st.info("👆 위에서 엑셀 파일을 업로드해주세요.")

        # 샘플 데이터 구조 안내
        st.header("📋 예상 데이터 구조")
        st.markdown("업로드할 엑셀 파일에는 다음 컬럼들이 포함되어야 합니다:")

        required_columns = [
            "계량기-당월지침",
            "비보정-당월지침",
            "전년동월팩터",
            "당월팩터",
            "전월팩터",
            "계량기-전월사용량",
            "계량기-전년동월사용량",
            "계량기-당월사용량",
        ]

        sample_df = pd.DataFrame({col: ["..."] for col in required_columns})
        st.dataframe(sample_df, use_container_width=True)


if __name__ == "__main__":
    main()
