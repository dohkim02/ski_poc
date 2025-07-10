import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
from tqdm import tqdm
import time


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
        print(f"행 처리 중 오류 발생: {e}")
        return ""


def process_excel_parallel(
    input_file,
    output_file=None,
    max_workers=None,
    uncorrected_threshold=30,
    factor_threshold=0.04,
):
    """
    엑셀 파일을 병렬로 처리하는 메인 함수

    Parameters:
    input_file (str): 입력 엑셀 파일 경로
    output_file (str): 출력 엑셀 파일 경로 (None이면 자동 생성)
    max_workers (int): 병렬 처리 워커 수 (None이면 CPU 코어 수)
    uncorrected_threshold (int): 비보정지침 이상 임계값 (기본값: 30)
    factor_threshold (float): 팩터 차이 임계값 (기본값: 0.04, 즉 4%)
    """

    print(f"📁 엑셀 파일 읽는 중: {input_file}")
    print(
        f"⚙️  설정값 - 비보정지침 임계값: {uncorrected_threshold}, 팩터 임계값: {factor_threshold*100:.1f}%"
    )

    # 엑셀 파일 읽기
    try:
        with tqdm(desc="파일 로딩", unit="MB") as pbar:
            df = pd.read_excel(input_file)
            pbar.update(1)
        print(f"✅ 데이터 로드 완료: {len(df):,}행, {len(df.columns)}컬럼")
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return

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
        print(f"⚠️  경고: 다음 컬럼들이 없습니다: {missing_columns}")
        print(f"📋 사용 가능한 컬럼들: {list(df.columns)}")

    # 데이터를 딕셔너리 형태로 변환
    print("🔄 데이터 변환 중...")
    with tqdm(desc="데이터 변환", total=len(df)) as pbar:
        row_data_list = df.to_dict("records")
        pbar.update(len(df))

    # 병렬 처리
    total_rows = len(row_data_list)
    max_workers = max_workers or os.cpu_count()
    print(f"🚀 병렬 처리 시작 (워커 수: {max_workers}, 총 {total_rows:,}행)")

    results = []
    completed_count = 0
    start_time = time.time()

    # 진행률 표시를 위한 tqdm 생성
    with tqdm(
        total=total_rows,
        desc="데이터 처리",
        unit="행",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:

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
                    pbar.update(1)

                    # 현재 시간과 속도 계산
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        rate = completed_count / elapsed_time
                        pbar.set_postfix(속도=f"{rate:.1f}행/초")

                except Exception as e:
                    print(f"\n⚠️  행 {index} 처리 중 오류: {e}")
                    results.append((index, ""))
                    completed_count += 1
                    pbar.update(1)

    # 결과를 원래 순서대로 정렬
    print("📊 결과 정렬 중...")
    with tqdm(desc="결과 정렬", total=len(results)) as pbar:
        results.sort(key=lambda x: x[0])
        pbar.update(len(results))

    flags = [result[1] for result in results]

    # 새로운 컬럼 추가
    print("📝 결과 컬럼 추가 중...")
    with tqdm(desc="컬럼 추가", total=1) as pbar:
        df["이상치_플래그"] = flags
        pbar.update(1)

    # 출력 파일명 생성
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.xlsx"

    # 결과 저장
    print(f"💾 결과 저장 중: {output_file}")
    try:
        with tqdm(desc="파일 저장", unit="MB") as pbar:
            df.to_excel(output_file, index=False)
            pbar.update(1)

        # 처리 시간 계산
        total_time = time.time() - start_time
        print(f"✅ 처리 완료! 결과가 {output_file}에 저장되었습니다.")
        print(f"⏱️  총 처리 시간: {total_time:.2f}초")
        print(f"📈 평균 처리 속도: {total_rows/total_time:.1f}행/초")

        # 이상치 통계
        flag_counts = df["이상치_플래그"].value_counts()
        print("\n" + "=" * 50)
        print("📊 이상치 탐지 결과")
        print("=" * 50)
        print(f"📋 총 행 수: {len(df):,}")
        print(f"🚨 이상치 발견 행 수: {len(df[df['이상치_플래그'] != '']):,}")
        print(f"✅ 정상 행 수: {len(df[df['이상치_플래그'] == '']):,}")

        # 이상치 발견률 계산
        abnormal_rate = (len(df[df["이상치_플래그"] != ""]) / len(df)) * 100
        print(f"📊 이상치 발견률: {abnormal_rate:.2f}%")

        print("\n🔍 각 이상치 유형별 개수:")
        for flag, count in flag_counts.items():
            if flag != "":
                percentage = (count / len(df)) * 100
                print(f"  • {flag}: {count:,}개 ({percentage:.2f}%)")

    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")


def main():
    """메인 실행 함수"""
    # 사용자 설정값
    input_file = "./data/data.xlsx"  # 입력 파일 경로를 여기에 지정
    output_file = "output_processed.xlsx"  # 출력 파일 경로 (선택사항)

    # 임계값 설정 (사용자가 원하는 값으로 수정)
    uncorrected_threshold = 30  # 비보정지침 이상 임계값
    factor_threshold = 0.04  # 팩터 차이 임계값 (4% = 0.04)

    print("🎯 엑셀 파일 처리 설정")
    print("-" * 50)
    print(f"📂 입력 파일: {input_file}")
    print(f"📁 출력 파일: {output_file}")
    print(f"⚙️  비보정지침 임계값: {uncorrected_threshold}")
    print(f"⚙️  팩터 임계값: {factor_threshold*100:.1f}%")
    print("-" * 50)

    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        print("💡 input_file 변수에 올바른 파일 경로를 지정해주세요.")
        return

    # 파일 크기 확인
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"📊 파일 크기: {file_size:.2f} MB")

    # 처리 실행
    print("🚀 엑셀 파일 처리를 시작합니다...")

    process_excel_parallel(
        input_file,
        output_file,
        max_workers=4,
        uncorrected_threshold=uncorrected_threshold,
        factor_threshold=factor_threshold,
    )

    print("\n🎉 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()
