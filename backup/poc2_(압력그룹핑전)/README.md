# 🔥 가스 사용량 이상 데이터 분석 시스템

업종별 가스 사용량 패턴을 분석하여 이상 데이터를 탐지하는 Streamlit 웹 애플리케이션입니다.

## ✨ 주요 기능

- 📊 **기준 데이터 시각화**: 업종별, 열량별 월별 사용량 기준 데이터를 테이블과 차트로 확인
- 🔍 **이상 데이터 분석**: 파일 업로드를 통한 자동 이상 데이터 탐지
- 📈 **분석 결과 확인**: 상세한 분석 결과와 이상 사유 제공
- 💾 **결과 다운로드**: 분석 결과를 TXT 파일로 다운로드

## 🚀 설치 및 실행

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

#### 방법 1: 실행 스크립트 사용 (권장)

```bash
python run_streamlit.py
```

#### 방법 2: 직접 Streamlit 실행

```bash
streamlit run streamlit_app.py
```

### 3. 웹 브라우저 접속

실행 후 브라우저에서 다음 주소로 접속하세요:

```
http://localhost:8501
```

## 📱 사용법

### 1. 기준 데이터 시각화 탭

- **업종별 기준 데이터**:

  - 드롭다운에서 업종을 선택하여 해당 업종의 월별 사용량 패턴 확인
  - Median과 IQR 값을 테이블과 차트로 시각화

- **열량별 기준 데이터**:
  - 열량 구간별 사용량 패턴 확인
  - 선택한 열량 구간의 월별 Median/IQR 데이터 시각화

### 2. 이상 데이터 분석 탭

1. **파일 업로드**: `preprocessed.txt` 파일을 드래그&드롭 또는 버튼을 통해 업로드
2. **데이터 미리보기**: 업로드된 데이터의 구조 확인 (선택사항)
3. **분석 시작**: "🚀 이상 데이터 분석 시작" 버튼 클릭
4. **진행률 확인**: 실시간 분석 진행률 모니터링

### 3. 분석 결과 탭

- **결과 요약**: 전체 데이터 수, 이상 데이터 수, 이상률 확인
- **상세 결과**: 각 이상 사례의 상세 정보 및 사유 확인
- **결과 다운로드**: TXT 파일로 분석 결과 다운로드

## 📁 파일 구조

```
poc2/
├── streamlit_app.py          # 메인 Streamlit 애플리케이션
├── run_streamlit.py          # 실행 스크립트
├── requirements.txt          # 필수 라이브러리 목록
├── _run.py                   # 분석 로직
├── utils.py                  # 유틸리티 함수들
├── prompt.py                 # LLM 프롬프트 템플릿
├── preprocessed.txt          # 샘플 데이터 파일
├── make_instruction/
│   ├── group_biz_with_usage.json    # 업종별 기준 데이터
│   └── group_heat_input.xlsx        # 열량별 기준 데이터
└── README.md                 # 이 파일
```

## 🔧 설정

### 환경 변수

LLM 모델 사용을 위해 필요한 API 키를 환경 변수로 설정하세요:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 데이터 파일 경로

기본적으로 다음 경로의 파일들을 사용합니다:

- 업종별 기준 데이터: `./make_instruction/group_biz_with_usage.json`
- 열량별 기준 데이터: `./make_instruction/group_heat_input.xlsx`

## 📊 입력 데이터 형식

업로드할 `preprocessed.txt` 파일은 다음 형식이어야 합니다:

```
업태: 서비스업
업종: 일반음식점
용도: 업무난방 전용
보일러 열량: 50000
연소기 열량: 0
1월: 1234.5
2월: 1456.7
...
12월: 987.3

업태: 제조업
...
```

## ⚠️ 주의사항

1. **대용량 파일**: 데이터가 많을 경우 분석 시간이 오래 걸릴 수 있습니다.
2. **동시 실행**: 한 번에 하나의 분석만 실행하는 것을 권장합니다.
3. **메모리 사용량**: 대량 데이터 처리 시 충분한 메모리가 필요합니다.

## 🐛 문제 해결

### 일반적인 문제들

1. **모듈을 찾을 수 없다는 오류**:

   ```bash
   pip install -r requirements.txt
   ```

2. **포트 충돌**:
   다른 포트로 실행하려면:

   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

3. **권한 오류**:
   실행 권한 부여:
   ```bash
   chmod +x run_streamlit.py
   ```

## 📞 지원

문제가 발생하거나 개선 사항이 있으시면 개발팀에 문의해주세요.

---

**Made with ❤️ using Streamlit and LangChain**
