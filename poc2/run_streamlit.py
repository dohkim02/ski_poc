#!/usr/bin/env python3
"""
Streamlit 앱 실행 스크립트
"""

import subprocess
import sys
import os


def main():
    """Streamlit 앱을 실행합니다."""

    # 현재 스크립트의 디렉토리로 이동
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("🔥 가스 사용량 이상 데이터 분석 시스템을 시작합니다...")
    print(f"📁 작업 디렉토리: {script_dir}")
    print("🌐 브라우저에서 http://localhost:8501 로 접속하세요.")
    print("=" * 50)

    # Streamlit 앱 실행
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=false",
                "--browser.gatherUsageStats=false",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit 실행 중 오류 발생: {e}")
        print("💡 다음 명령어로 수동 실행을 시도해보세요:")
        print("   streamlit run streamlit_app.py")
    except KeyboardInterrupt:
        print("\n👋 Streamlit 앱이 종료되었습니다.")


if __name__ == "__main__":
    main()
