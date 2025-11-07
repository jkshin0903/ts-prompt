# ts-prompt

LLM을 활용한 시계열 데이터 예측 파이프라인. CSV 형식의 시계열 데이터를 행(row) 단위로 처리하여 LLM(GPT/Gemini)에 프롬프트로 전달하고, 다음 시계열 값을 예측합니다.

## 권장 환경
- Python 3.11+ (LangChain/Pydantic 호환 이슈로 3.14 비권장)
- 패키지 설치: `pip install -r requirements.txt`
- API 키(환경변수):
  - GPT(OpenAI): `OPENAI_API_KEY`
  - Gemini(Google): `GOOGLE_API_KEY`

## 프로젝트 구조
```
./
├─ dataset/                    # 시계열 데이터셋 (CSV 파일)
│  ├─ ETT-small/              # ETT 벤치마크 데이터셋
│  └─ ...                     # 기타 데이터셋
├─ templates/
│  └─ instruction_m2n.txt     # M2N 예측 프롬프트 템플릿
├─ utils/
│  ├─ dataset.py              # CSV 파일 처리 및 정렬
│  ├─ rows.py                 # 행 단위 데이터 로드/파싱 유틸
│  └─ prompt.py               # 프롬프트 생성 및 저장
├─ responses/                 # LLM 응답 및 메트릭 저장 위치
├─ scripts/
│  ├─ run_forecast_m2n.sh    # M2N 예측 실행 스크립트
│  └─ evaluate_m2n.sh         # 예측 결과 평가 스크립트
└─ src/
   ├─ forecast_m2n.py         # M2N 예측 실행 로직
   └─ evaluate_m2n.py         # MSE/MAE 평가 로직
```

## 주요 모듈 요약

### `utils/rows.py`
- CSV 파일에서 시계열 행 데이터를 로드
- 날짜/타임스탬프 컬럼 자동 감지
- 행 데이터 파싱 및 포맷팅

### `utils/prompt.py`
- M2N 예측 프롬프트 생성 (`create_m2n_prompt`)
- 입력 행 M개를 기반으로 다음 N개 행 예측 요청

### `src/forecast_m2n.py`
- CSV 파일에서 행 데이터 로드
- 프롬프트 생성 및 LLM 호출
- 응답 및 프롬프트 저장

### `src/evaluate_m2n.py`
- 예측 결과와 실제 값 비교
- MSE(Mean Squared Error) 및 MAE(Mean Absolute Error) 계산
- 메트릭을 JSON 형식으로 저장

## 사용 방법

### 1. 예측 실행

```bash
# 기본 사용법
./scripts/run_forecast_m2n.sh <csv_file> [model_name]

# 예시
./scripts/run_forecast_m2n.sh dataset/ETT-small/ETTh1.csv gpt-4.1-mini-2025-04-14
```

스크립트는 여러 조합(num_input/num_predict)으로 자동 실행합니다:
- 24/24, 36/36, 48/48, 60/60, 72/72, 84/84, 96/96

결과는 `responses/{dataset_name}/{model_dir}_{num_input}_{num_predict}/`에 저장됩니다.

### 2. 평가 실행

```bash
# 기본 사용법
./scripts/evaluate_m2n.sh <csv_file> [model_name]

# 예시
./scripts/evaluate_m2n.sh dataset/ETT-small/ETTh1.csv gpt-4.1-mini-2025-04-14
```

스크립트는 예측 결과를 자동으로 찾아 평가하고, 메트릭을 `metrics.json`에 저장합니다.
응답 파일이 없으면 자동으로 예측을 실행합니다.

### 3. 직접 Python 스크립트 실행

```bash
# 예측 실행
python src/forecast_m2n.py \
  --csv dataset/ETT-small/ETTh1.csv \
  --num_input 30 \
  --num_predict 30 \
  --model gpt-4.1-mini-2025-04-14 \
  --start_index 0 \
  --output_dir responses/ETTh1/test

# 평가 실행
python src/evaluate_m2n.py \
  --csv dataset/ETT-small/ETTh1.csv \
  --num_input 30 \
  --num_predict 30 \
  --model gpt-4.1-mini-2025-04-14 \
  --start_index 0 \
  --auto_response
```

## 데이터 형식

### CSV 파일 요구사항
- 첫 번째 컬럼: 날짜/타임스탬프 (컬럼명: `date`, `Date`, `time`, `timestamp` 등)
- 나머지 컬럼: 숫자형 특징 값들
- 예시:
  ```csv
  date,feature1,feature2,feature3
  2020-01-01,1.5,2.3,3.1
  2020-01-02,1.6,2.4,3.2
  ```

### 응답 형식
LLM은 다음 형식으로 응답해야 합니다:
```
2020-01-03,1.7,2.5,3.3
2020-01-04,1.8,2.6,3.4
...
```

## 출력 파일 구조

```
responses/
└─ {dataset_name}/
   └─ {model_name}_{num_input}_{num_predict}/
      ├─ {dataset_name}_prompt.txt          # 사용된 프롬프트
      ├─ {dataset_name}_response_{model}.txt # LLM 응답
      └─ metrics.json                        # 평가 메트릭 (MSE, MAE)
```

## 자주 묻는 질문(FAQ)

- **왜 행(row) 단위로 처리하나요?**
  - Patch 단위 예측이 모델에 부담을 주어 행 단위로 전환했습니다. 각 행은 하나의 시점을 나타냅니다.

- **`ModuleNotFoundError: No module named 'src'`?**
  - 프로젝트 루트에서 스크립트를 실행하세요. 쉘 스크립트는 자동으로 프로젝트 루트로 이동합니다.

- **Python 3.14에서 Pydantic V1 경고가 나옵니다.**
  - LangChain/Pydantic 호환 이슈입니다. Python 3.11 사용을 권장합니다.

- **어떤 데이터셋을 사용할 수 있나요?**
  - 날짜/타임스탬프 컬럼과 숫자형 특징 컬럼을 가진 모든 CSV 파일을 사용할 수 있습니다. ETT-small, M4 등 벤치마크 데이터셋을 지원합니다.

## 라이선스
- 프로젝트 루트의 LICENSE(있는 경우)를 참조하세요.
