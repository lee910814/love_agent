"""
대화 기억 모듈
- 단기 기억: 현재 세션 대화 히스토리
- 장기 기억: 사용자 정보, 감정 키워드, 중요 사건 저장
"""

import json
import os
from datetime import datetime
from pathlib import Path


class Memory:
    def __init__(self, user_id: str, memory_dir: str = "./memory"):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / f"{user_id}.json"

        # 단기 기억 (현재 세션)
        self.short_term: list[dict] = []

        # 장기 기억 (파일 persistent)
        self.long_term = self._load_long_term()

    # ── 단기 기억 ─────────────────────────────────────────

    def add_turn(self, role: str, content: str):
        """대화 턴 추가"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_recent(self, n: int = 10) -> list[dict]:
        """최근 n턴 반환 (role/content만)"""
        recent = self.short_term[-n:]
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def clear_session(self):
        self.short_term = []

    # ── 장기 기억 ─────────────────────────────────────────

    def _load_long_term(self) -> dict:
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_name": None,
            "user_info": [],       # ["직장인", "고양이 좋아함", ...]
            "emotional_keywords": [],  # 자주 언급한 감정/상황
            "important_events": [],    # 중요한 고백, 약속 등
            "created_at": datetime.now().isoformat(),
        }

    def save_long_term(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, ensure_ascii=False, indent=2)

    def update_user_name(self, name: str):
        self.long_term["user_name"] = name
        self.save_long_term()

    def add_user_info(self, info: str):
        if info not in self.long_term["user_info"]:
            self.long_term["user_info"].append(info)
            self.save_long_term()

    def add_important_event(self, event: str):
        self.long_term["important_events"].append({
            "event": event,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        self.save_long_term()

    def get_context_summary(self) -> str:
        """장기 기억을 system prompt용 텍스트로 변환"""
        lt = self.long_term
        lines = []

        if lt["user_name"]:
            lines.append(f"상대방 이름: {lt['user_name']}")

        if lt["user_info"]:
            lines.append(f"알고 있는 정보: {', '.join(lt['user_info'])}")

        if lt["important_events"]:
            events = [f"[{e['date']}] {e['event']}" for e in lt["important_events"][-5:]]
            lines.append(f"중요한 순간들:\n" + "\n".join(events))

        return "\n".join(lines) if lines else ""
