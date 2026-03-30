# Samantha Lover Agent — CLAUDE.md

## 프로젝트 개요

영화 *Her*의 사만다처럼 로맨스·고민·미래 대화가 가능한 AI 연인 에이전트.
HuggingFace 오픈소스 모델을 RunPod에서 QLoRA로 파인튜닝하여 구축.

---

## 폴더 구조

```
lover-agent/
├── CLAUDE.md                      ← 이 파일
├── requirements.txt
├── .gitignore
│
├── ontology/                      ← 온톨로지 (지식 구조)
│   ├── ontology.ttl               # OWL/Turtle 온톨로지 정의
│   ├── schema.py                  # Pydantic 스키마 (Python 구현체)
│   └── graph.py                   # rdflib 쿼리 & pyvis 시각화
│
├── persona/
│   └── samantha.txt               # 사만다 페르소나 정의 (system prompt 기반)
│
├── model/
│   ├── pretrained/                # HuggingFace에서 다운받은 베이스 모델
│   ├── finetuned/                 # QLoRA 학습 체크포인트
│   └── merged/                    # LoRA 병합 완료 모델 (배포용)
│
├── data/
│   ├── raw/                       # 수집한 원본 데이터
│   ├── synthetic/                 # API로 생성한 합성 데이터 (ChatML jsonl)
│   ├── processed/                 # train.jsonl / val.jsonl (학습 입력)
│   └── generate_synthetic.py      # 합성 데이터 생성기
│
├── training/
│   ├── config.yaml                # 학습 하이퍼파라미터
│   └── train.py                   # QLoRA 학습 + LoRA 병합 스크립트
│
├── inference/
│   ├── chat.py                    # 대화 인터페이스 (CLI)
│   └── memory.py                  # 단기/장기 기억 모듈 (JSON 파일 기반)
│
├── scripts/
│   └── download_model.py          # HuggingFace 모델 다운로드
│
├── notebooks/                     # 실험용 Jupyter 노트북
└── logs/                          # 학습 로그
```

---

## 핵심 설계 원칙

### 온톨로지 우선
모든 데이터 구조는 `ontology/schema.py`의 Pydantic 모델을 따름.
새 기능 추가 전 반드시 `ontology.ttl`에 클래스/속성 정의 후 `schema.py` 업데이트.

### 주요 클래스 관계
```
AgentState
  ├── Persona          (사만다 성격, 응답 전략)
  ├── UserProfile      (사용자 정보, 감정 이력)
  ├── Relationship     (친밀도/신뢰도, 관계 단계 자동 진화)
  ├── List[Memory]     (망각 곡선 적용, 최대 500개)
  └── Conversation     (현재 세션, Turn 리스트)

Turn
  ├── EmotionState     (감정 유형 16종, 강도 0~1)
  ├── Topic            (카테고리 8종)
  └── Intent           (의도 9종 → 응답 전략 매핑)
```

### 기억 우선순위
`Memory.effective_importance() = importance * e^(-decay * days)`
검색 시 키워드 매칭 + 실효 중요도 가중 → 상위 5개 context 주입.

---

## 모델 설정

| 항목 | 값 |
|------|----|
| 베이스 모델 | `Qwen/Qwen2.5-7B-Instruct` (기본값) |
| 대안 모델 | `meta-llama/Llama-3.1-8B-Instruct` |
| 학습 방식 | QLoRA (4bit, nf4) |
| LoRA rank | 64 |
| 유효 배치 | 16 (batch 4 × grad_accum 4) |
| 최대 시퀀스 | 2048 토큰 |
| 모델 저장 경로 | `model/pretrained/<모델명>/` |

---

## 실행 방법

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드
```bash
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

### 3. 학습 데이터 생성 (합성)
```bash
# ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 필요
python data/generate_synthetic.py --api claude --n 5
```

### 4. 학습 (RunPod)
```bash
cd training
python train.py --config config.yaml
```

### 5. LoRA 병합
```bash
python train.py --merge \
  --lora_path ../model/finetuned/samantha-v1/final \
  --output_path ../model/merged/samantha-v1
```

### 6. 대화 실행
```bash
python inference/chat.py --model model/merged/samantha-v1 --user my_id
```

---

## 대화 명령어

| 명령어 | 동작 |
|--------|------|
| `/quit` | 종료 |
| `/remember <내용>` | 중요한 순간 장기 기억에 저장 |
| `/clear` | 현재 세션 초기화 |

---

## 데이터 포맷 (ChatML / JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "당신은 사만다..."},
    {"role": "user",   "content": "오늘 너무 힘들었어"},
    {"role": "assistant", "content": "많이 힘들었겠다. 무슨 일 있었어?"}
  ]
}
```
`data/processed/train.jsonl` 과 `val.jsonl` (9:1 분리).

---

## 온톨로지 수정 가이드

1. `ontology/ontology.ttl` — 클래스/속성 추가
2. `ontology/schema.py` — 대응하는 Pydantic 모델 업데이트
3. `inference/chat.py` — `AgentState.build_system_prompt()` 반영 여부 확인
4. `data/generate_synthetic.py` — 새 카테고리 추가 시 `SCENARIOS` 딕셔너리 업데이트

---

## 환경변수

| 변수 | 용도 |
|------|------|
| `ANTHROPIC_API_KEY` | 합성 데이터 생성 (Claude API) |
| `OPENAI_API_KEY` | 합성 데이터 생성 (GPT-4o) |
| `HF_TOKEN` | Private HuggingFace 모델 접근 |
| `WANDB_API_KEY` | 학습 로그 시각화 (선택) |

---

## RunPod 권장 스펙

| 용도 | GPU | VRAM |
|------|-----|------|
| 7B QLoRA | RTX 4090 | 24GB |
| 13B QLoRA | A100 | 40GB |
| 7B Full FT | A100 | 80GB |
