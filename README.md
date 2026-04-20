# GroundWire

## 프로젝트 개요

GroundWire는 국내외 주요 뉴스를 자동으로 수집하고, 비슷한 기사를 하나의 이슈로 묶은 뒤, 근거 기반 검증과 LLM 분석을 거쳐 웹 대시보드에 보여주는 `trust-first` 뉴스 분석 프로젝트입니다.

이 프로젝트의 목표는 다음 과정을 안정적으로 자동화하는 것입니다.

- 뉴스 수집
- 기사 정제
- 이슈 클러스터링
- 근거 추출 및 신뢰도 계산
- LLM 기반 요약/포인트/claim 생성
- `READY` / `HOLD` 판정
- 웹 대시보드 제공

## 활용 기술

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5.4--mini-412991?logo=openai&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Storage-003B57?logo=sqlite&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-Templates-B41717?logo=jinja&logoColor=white)
![APScheduler](https://img.shields.io/badge/APScheduler-Scheduler-4B5563)

## 전체적인 개발 플로우

전체 흐름은 아래 순서로 구성됩니다.

1. 공개 RSS 기반으로 뉴스 기사를 수집합니다.
2. 전처리를 통해 중복, 노이즈 텍스트, 품질이 낮은 기사를 정리합니다.
3. 제목/본문 키워드, 숫자 정보, 시간 정보를 기준으로 유사 기사를 클러스터링합니다.
4. 각 클러스터에서 근거 스니펫을 추출하고, 출처 다양성/최신성/교차 확인 정도를 기반으로 신뢰도를 계산합니다.
5. 백엔드에서만 LLM을 호출해 summary, key points, atomic claims를 생성합니다.
6. grounding 검증을 통해 각 claim이 실제 근거로 얼마나 지지되는지 확인합니다.
7. 충분히 검증된 이슈는 `READY`, 애매한 이슈는 `HOLD`로 분류합니다.
8. 최종 결과를 SQLite에 저장하고, 웹 대시보드/API로 제공합니다.

## 핵심 로직

이 프로젝트의 핵심 로직은 아래처럼 간단히 정리할 수 있습니다.

1. 뉴스를 자동으로 수집하고 전처리합니다.
2. 비슷한 기사들을 하나의 이슈로 클러스터링합니다.
3. 근거 스니펫과 신뢰도 점수를 바탕으로 LLM이 summary, key points, claims를 생성합니다.
4. grounding 검증을 거쳐 충분히 검증된 이슈는 `READY`, 그렇지 않은 이슈는 `HOLD`로 분류합니다.

## 설치 및 실행 방법

### 1. 코드 받기

```bash
git clone <repository-url>
cd New\ project
```

### 2. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 의존성 설치

`requirements.txt`를 사용하는 경우:

```bash
pip install -r requirements.txt
```

패키지 설치 방식으로 사용하는 경우:

```bash
pip install -e .
```

### 4. `.env` 파일 설정

실행 전에 프로젝트 루트에 `.env` 파일을 만들고 아래처럼 값을 넣어주면 됩니다.

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.4-mini
EMBEDDING_MODEL=text-embedding-3-small
```

`OPENAI_API_KEY`에는 본인 키를 넣으면 되고, 모델 값은 필요에 따라 변경할 수 있습니다.

### 5. 서버 실행

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

실행 후 접속 주소:

- Dashboard: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

## 참고

- 프론트엔드는 직접 LLM API를 호출하지 않습니다.
- LLM 호출은 모두 백엔드에서만 수행됩니다.
- 기본 보고 채널은 웹 대시보드입니다.
- 배포는 필수가 아니며, 로컬 실행 기준으로 동작하도록 설계되어 있습니다.
