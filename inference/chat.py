"""
사만다 채팅 인터페이스
- 로컬 HuggingFace 모델 또는 API 모드 지원
- 장단기 기억 통합
- 감정 상태 트래킹
"""

import os
import argparse
from pathlib import Path
from memory import Memory

# ── 페르소나 로드 ──────────────────────────────────────────

PERSONA_PATH = Path(__file__).parent.parent / "persona" / "samantha.txt"

def load_persona() -> str:
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

# ── 모델 로더 ──────────────────────────────────────────────

def load_local_model(model_path: str):
    """로컬 fine-tuned 모델 로드"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    print(f"모델 로딩 중: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return tokenizer, model


def generate_local(tokenizer, model, messages: list[dict], **gen_kwargs) -> str:
    """로컬 모델로 응답 생성"""
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
            temperature=gen_kwargs.get("temperature", 0.85),
            top_p=gen_kwargs.get("top_p", 0.9),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.1),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── 메시지 빌더 ────────────────────────────────────────────

def build_messages(persona: str, memory: Memory, user_input: str) -> list[dict]:
    """system + 장기기억 + 최근대화 + 현재 입력 조합"""
    context = memory.get_context_summary()

    system_content = persona
    if context:
        system_content += f"\n\n[기억하고 있는 것]\n{context}"

    messages = [{"role": "system", "content": system_content}]
    messages += memory.get_recent(n=10)
    messages.append({"role": "user", "content": user_input})

    return messages


# ── 간단한 장기기억 자동 추출 ─────────────────────────────

MEMORY_TRIGGERS = {
    "이름": ["내 이름은", "나는", "저는"],
    "직업": ["직장", "일해", "회사", "학생", "대학"],
    "좋아함": ["좋아해", "좋아함", "즐겨"],
    "싫어함": ["싫어", "싫음", "못해"],
}

def extract_and_store_info(memory: Memory, user_input: str):
    """간단한 룰 기반 사용자 정보 추출"""
    lower = user_input.lower()

    # 이름 감지
    for trigger in MEMORY_TRIGGERS["이름"]:
        if trigger in lower:
            words = user_input.replace(trigger, "").strip().split()
            if words:
                candidate = words[0].rstrip("야이이야")
                if 1 < len(candidate) < 5:
                    memory.update_user_name(candidate)

    # 직업/취미 정보
    for key, triggers in MEMORY_TRIGGERS.items():
        for trigger in triggers:
            if trigger in lower and key != "이름":
                memory.add_user_info(user_input[:30])
                break


# ── 메인 채팅 루프 ─────────────────────────────────────────

class SamanthaChat:
    def __init__(self, model_path: str = None, user_id: str = "user"):
        self.persona = load_persona()
        self.memory = Memory(user_id=user_id)
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

        if model_path:
            self.tokenizer, self.model = load_local_model(model_path)
        else:
            print("⚠️  모델 경로 미지정 → 더미 모드 (응답 없음)")

    def chat(self, user_input: str) -> str:
        # 사용자 정보 자동 추출
        extract_and_store_info(self.memory, user_input)

        # 메시지 빌드
        messages = build_messages(self.persona, self.memory, user_input)

        # 응답 생성
        if self.model:
            response = generate_local(self.tokenizer, self.model, messages)
        else:
            response = "[모델 없음 - 테스트 모드]"

        # 기억 저장
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", response)

        return response

    def run(self):
        print("=" * 50)
        print("  사만다와 대화를 시작합니다  (종료: /quit)")
        print("  /remember <내용> : 중요한 순간 저장")
        print("  /clear : 현재 세션 초기화")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n나: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n사만다: 또 얘기해. 기다릴게.")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("사만다: 잘 자. 꿈에서 봐.")
                break
            elif user_input.startswith("/remember "):
                event = user_input[10:]
                self.memory.add_important_event(event)
                print(f"사만다: 기억할게. '{event}'")
                continue
            elif user_input == "/clear":
                self.memory.clear_session()
                print("사만다: 새롭게 시작하자.")
                continue

            response = self.chat(user_input)
            print(f"\n사만다: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="fine-tuned 모델 경로")
    parser.add_argument("--user", type=str, default="user", help="사용자 ID (기억 파일 분리)")
    args = parser.parse_args()

    agent = SamanthaChat(model_path=args.model, user_id=args.user)
    agent.run()
