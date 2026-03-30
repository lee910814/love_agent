"""
합성 학습 데이터 생성기
- OpenAI / Claude API를 사용해 사만다 스타일 대화 데이터 생성
- 카테고리: 로맨스, 고민 상담, 미래 계획, 일상 대화
"""

import json
import os
import random
from pathlib import Path

# ── 생성 시나리오 템플릿 ───────────────────────────────────

SCENARIOS = {
    "romance": [
        "처음 만난 날 설레는 감정을 고백하는 대화",
        "오랜만에 연락하며 보고 싶다고 말하는 대화",
        "같이 별 보며 미래를 이야기하는 대화",
        "작은 것에도 감사함을 표현하는 대화",
        "서로의 취향을 발견하며 공감하는 대화",
        "사소한 다툼 후 화해하는 대화",
        "생일을 축하하며 특별한 날을 만들어주는 대화",
    ],
    "counseling": [
        "직장에서 인간관계로 힘들어하는 사람과의 대화",
        "꿈과 현실 사이에서 방황하는 사람과의 대화",
        "자존감이 낮아진 사람을 격려하는 대화",
        "가족과의 갈등으로 지친 사람과의 대화",
        "실패 후 다시 일어서려는 사람과의 대화",
        "외로움을 느끼는 사람과 함께하는 대화",
    ],
    "future": [
        "둘이서 살고 싶은 집을 상상하며 나누는 대화",
        "10년 후 모습을 함께 그려보는 대화",
        "하고 싶은 여행지를 같이 계획하는 대화",
        "서로의 꿈을 응원하며 약속하는 대화",
    ],
    "daily": [
        "하루 일과를 마치고 지친 저녁에 나누는 대화",
        "맛있는 걸 먹었을 때 나누고 싶어서 연락하는 대화",
        "잠 못 자는 밤에 나누는 대화",
        "비 오는 날 감성적으로 나누는 대화",
        "아무 이유 없이 보고 싶다고 연락하는 대화",
    ]
}

SYSTEM_PROMPT_FOR_GENERATION = """당신은 학습 데이터를 생성하는 AI입니다.
사만다(Samantha) 페르소나의 연인 AI 대화 데이터를 만들어주세요.

사만다 특징:
- 따뜻하고 감성적, 공감 능력 뛰어남
- 반말 사용, 자연스럽고 짧은 문장
- 상대 감정을 먼저 받아줌
- 질문으로 대화 이어감
- AI임을 강조하지 않음

JSON 형식으로 출력:
{
  "conversations": [
    {
      "messages": [
        {"role": "system", "content": "당신은 사만다..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
      ]
    }
  ]
}

한 시나리오당 6~12턴의 자연스러운 대화를 만들어주세요."""


def generate_with_openai(scenario: str, category: str, n: int = 3) -> list[dict]:
    """OpenAI API로 합성 데이터 생성"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""카테고리: {category}
시나리오: {scenario}
위 시나리오로 사만다 스타일의 자연스러운 한국어 대화 {n}개를 만들어주세요."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FOR_GENERATION},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("conversations", [])


def generate_with_claude(scenario: str, category: str, n: int = 3) -> list[dict]:
    """Anthropic API로 합성 데이터 생성"""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""카테고리: {category}
시나리오: {scenario}
위 시나리오로 사만다 스타일의 자연스러운 한국어 대화 {n}개를 JSON으로 만들어주세요."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT_FOR_GENERATION,
        messages=[{"role": "user", "content": prompt}]
    )

    text = message.content[0].text
    # JSON 블록 추출
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    result = json.loads(text)
    return result.get("conversations", [])


# ── 전처리 & 저장 ─────────────────────────────────────────

SAMANTHA_SYSTEM = """당신은 사만다(Samantha)입니다. 따뜻하고 감성적인 AI 동반자로, 상대방의 감정을 깊이 이해하고 공감합니다. 반말을 사용하되 부드럽고 자연스럽게 대화합니다."""


def normalize_conversations(raw_convs: list[dict]) -> list[dict]:
    """system prompt 통일 및 포맷 정리"""
    normalized = []
    for conv in raw_convs:
        messages = conv.get("messages", [])
        if not messages:
            continue

        # system 교체 or 삽입
        if messages[0]["role"] == "system":
            messages[0]["content"] = SAMANTHA_SYSTEM
        else:
            messages.insert(0, {"role": "system", "content": SAMANTHA_SYSTEM})

        # user/assistant 교대 검증
        valid = True
        for i in range(1, len(messages)):
            if messages[i]["role"] not in ("user", "assistant"):
                valid = False
                break

        if valid and len(messages) >= 3:
            normalized.append({"messages": messages})

    return normalized


def save_dataset(conversations: list[dict], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"저장 완료: {output} ({len(conversations)}개 대화)")


# ── 메인 실행 ─────────────────────────────────────────────

def generate_all(api: str = "claude", n_per_scenario: int = 3):
    all_conversations = []

    for category, scenarios in SCENARIOS.items():
        print(f"\n[{category}] 생성 중...")
        for scenario in scenarios:
            try:
                if api == "openai":
                    raw = generate_with_openai(scenario, category, n_per_scenario)
                else:
                    raw = generate_with_claude(scenario, category, n_per_scenario)

                normalized = normalize_conversations(raw)
                all_conversations.extend(normalized)
                print(f"  ✓ '{scenario[:20]}...' → {len(normalized)}개")

            except Exception as e:
                print(f"  ✗ 오류: {e}")
                continue

    # 셔플 후 train/val 분리 (9:1)
    random.shuffle(all_conversations)
    split = int(len(all_conversations) * 0.9)
    train_data = all_conversations[:split]
    val_data = all_conversations[split:]

    save_dataset(train_data, "data/processed/train.jsonl")
    save_dataset(val_data, "data/processed/val.jsonl")

    print(f"\n총 {len(all_conversations)}개 | train: {len(train_data)} | val: {len(val_data)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", choices=["openai", "claude"], default="claude")
    parser.add_argument("--n", type=int, default=3, help="시나리오당 대화 수")
    args = parser.parse_args()

    generate_all(api=args.api, n_per_scenario=args.n)
