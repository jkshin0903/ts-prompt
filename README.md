# ts-prompt

간단한 시계열 패치 생성·예측 파이프라인. 암호화폐(또는 일반 시계열) 일별 OHLC 데이터를 패치 단위로 생성하고, LLM(GPT/Gemini)에 프롬프트로 전달하여 다음 패치들을 예측합니다.

## 권장 환경
- Python 3.11 (LangChain/Pydantic 호환 이슈로 3.14 비권장)
- 패키지 설치: `pip install -r requirements.txt`
- API 키(환경변수):
  - GPT(OpenAI): `OPENAI_API_KEY`
  - Gemini(Google): `GOOGLE_API_KEY`

## 프로젝트 구조
```
./
├─ dataset/
│  ├─ original/             # 원본 CSV (Binance_{SYMBOL}_d.csv)
│  ├─ train/                # 분리된 훈련 CSV ({SYMBOL}.csv)
│  └─ test/                 # 분리된 테스트 CSV ({SYMBOL}.csv)
├─ patches/
│  ├─ train/                # 훈련 패치 텍스트 ({SYMBOL}_patches.txt)
│  └─ test/                 # 테스트 패치 텍스트 ({SYMBOL}_patches.txt)
├─ prompts/
│  ├─ patch_structure_kr.txt  # 패치 구조(한국어)
│  └─ patch_structure_en.txt  # 패치 구조(영문)
├─ responses/               # LLM 응답 및 사용된 프롬프트 저장 위치(실행 시 생성)
├─ scripts/
│  ├─ run_forecast.sh       # main.py 실행 래퍼(프로젝트 루트에서 실행)
│  └─ example.sh            # 예시(타 프레임워크 용도, 이 프로젝트 직접 실행과 무관)
└─ src/
   ├─ split_dataset.py      # original → (train/test) CSV 분리
   ├─ extract_patches.py    # CSV → 패치 생성 및 텍스트 저장 CLI
   ├─ instruct_forcasting.py# 프롬프트 생성 유틸(kr/en), 패치 로더
   └─ main.py               # 패치 로드→프롬프트 생성→LLM 호출→응답 저장
```

## 주요 모듈 요약
- `src/split_dataset.py`
  - 원본 CSV(`dataset/original`)를 Date 오름차순 정렬 후 7:3 비율로 `dataset/train`, `dataset/test`로 분리.
  - 원본 헤더가 2번째 줄부터 시작하는 포맷을 지원(`header=1`).

- `src/extract_patches.py`
  - CSV의 `Date,Open,High,Low,Close`를 사용해 슬라이딩 윈도우 패치를 생성.
  - CLI 인자: `--split {train|test}`, `--patch_size`, `--stride`, `--base_dir`, `--out_dir`.
  - 결과는 `patches/{split}/{SYMBOL}_patches.txt`에 저장. 각 패치는 "===== Patch i =====" 헤더 + OHLC 문자열 행으로 구성.

- `src/instruct_forcasting.py`
  - 패치 리스트를 입력 받아 예측용 프롬프트(kr/en)를 생성.
  - `start_index`가 없으면 기본적으로 "마지막 M개" 패치를 입력으로 사용(최신 구간 예측 시나리오). 비교를 위해 처음부터 사용하려면 `--start_index 0`을 지정.
  - `load_patches_from_txt()`로 `*_patches.txt`를 다시 메모리로 로드 가능.

- `src/main.py`
  - 패치 로드(텍스트 또는 CSV) → 프롬프트 생성 → LLM 호출 → 응답/프롬프트 저장까지 수행.
  - 인자(일부):
    - 입력 소스: `--patch_file` 또는 `--csv` (상대경로는 프로젝트 루트 기준)
    - 프롬프트: `--num_input`, `--num_predict`, `--start_index`, `--language {kr|en}`
    - 모델: `--model` (예: `gpt-4.1-mini-2025-04-14`, `gemini-2.0-flash`) — 모델명만으로 타입 자동 판별
    - 실행: `--temperature`, `--output_dir`, `--save_prompt`, `--restrict_to_prompt`
  - `--restrict_to_prompt` 사용 시 시스템 메시지로 외부 지식 사용 금지 지시를 추가.

## 데이터 준비
1) 원본 CSV 배치
- `dataset/original/Binance_{SYMBOL}_d.csv` 형식으로 배치

2) 학습/테스트 분리
```bash
python src/split_dataset.py
```
- `dataset/train/{SYMBOL}.csv`, `dataset/test/{SYMBOL}.csv` 생성

3) 패치 생성(텍스트)
```bash
# 예: train split, patch_size=16, stride=1, 출력은 patches/train/
python src/extract_patches.py --split train --patch_size 16 --stride 1
```

## 예측 실행(LLM)
1) 단건 실행(main.py)
```bash
# 패치 텍스트로부터 입력 M=3, 다음 N=2 예측 (처음 패치부터 시작)
python src/main.py \
  --patch_file patches/train/ADAUSDT_patches.txt \
  --num_input 3 --num_predict 2 --start_index 0 \
  --language kr \
  --model gpt-4.1-mini-2025-04-14 \
  --output_dir responses/train/gpt_4_1_mini_2025_04_14 --save_prompt
```

2) 배치 실행(scripts/run_forecast.sh)
```bash
# 스크립트 내부 기본값(DATASET_TYPE/SPLIT/MODEL/LANGUAGE)을 조정 후 실행
./scripts/run_forecast.sh
```
- 자동으로 여러 조합(num_input/num_predict)을 실행하고 `responses/`에 저장합니다.
- 스크립트는 프로젝트 루트에서 실행되도록 `cd` 처리되어 있습니다.

## 패치 파일 형식
- 파일: `patches/{split}/{SYMBOL}_patches.txt`
- 예시:
```
===== Patch 0 =====
YYYY-MM-DD,Open,High,Low,Close
...

===== Patch 1 =====
...
```

## 자주 묻는 질문(FAQ)
- 왜 기본으로 마지막 M개 패치를 입력으로 쓰나요?
  - 최신 구간을 기준으로 다음 패치를 예측하는 일반 워크플로를 반영한 기본값입니다. 초기부터 비교가 필요하면 `--start_index 0`을 지정하세요.

- `ModuleNotFoundError: No module named 'src'`?
  - `src/main.py`는 로컬 임포트를 사용합니다. 프로젝트 루트에서 `python src/main.py ...`로 실행하거나, `scripts/run_forecast.sh`를 사용하세요.

- Python 3.14에서 Pydantic V1 경고가 나옵니다.
  - LangChain/Pydantic 호환 이슈입니다. Python 3.11 사용을 권장합니다.

## 라이선스
- 프로젝트 루트의 LICENSE(있는 경우)를 참조하세요.

