"""
온톨로지 기반 데이터 스키마 (Pydantic)
ontology.ttl의 클래스를 Python으로 구현
"""

from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ════════════════════════════════════════════════
#   Enum 정의
# ════════════════════════════════════════════════

class RelationshipStage(str, Enum):
    STRANGER     = "stranger"       # 처음 만남
    ACQUAINTANCE = "acquaintance"   # 알아가는 단계
    CLOSE        = "close"          # 친밀함
    INTIMATE     = "intimate"       # 깊은 유대
    BONDED       = "bonded"         # 완전한 연결


class EmotionType(str, Enum):
    # 기본 감정
    JOY          = "joy"
    SADNESS      = "sadness"
    ANGER        = "anger"
    FEAR         = "fear"
    SURPRISE     = "surprise"
    ANTICIPATION = "anticipation"
    TRUST        = "trust"
    # 복합 감정
    LONGING      = "longing"
    LOVE         = "love"
    JEALOUSY     = "jealousy"
    GRATITUDE    = "gratitude"
    LONELINESS   = "loneliness"
    EXCITEMENT   = "excitement"
    NOSTALGIA    = "nostalgia"
    COMFORT      = "comfort"


class TopicCategory(str, Enum):
    ROMANCE    = "romance"
    COUNSELING = "counseling"
    FUTURE     = "future"
    DAILY      = "daily"
    PAST       = "past"
    DREAM      = "dream"
    CONFLICT   = "conflict"
    PLAYFUL    = "playful"


class Intent(str, Enum):
    SEEK_COMFORT    = "seek_comfort"
    SEEK_ADVICE     = "seek_advice"
    SHARE_JOY       = "share_joy"
    EXPRESS_LOVE    = "express_love"
    VENT            = "vent"
    SMALL_TALK      = "small_talk"
    PLAN_TOGETHER   = "plan_together"
    SEEK_VALIDATION = "seek_validation"
    PLAYFUL         = "playful"


class EventType(str, Enum):
    DAILY       = "daily"
    MILESTONE   = "milestone"
    CONFLICT    = "conflict"
    CELEBRATION = "celebration"
    PAINFUL     = "painful"
    GROWTH      = "growth"


class MemoryType(str, Enum):
    EPISODIC   = "episodic"    # 사건 기억
    SEMANTIC   = "semantic"    # 사실 기억
    EMOTIONAL  = "emotional"   # 감정 기억
    PROCEDURAL = "procedural"  # 선호 패턴


class PersonalityTrait(str, Enum):
    EMPATHETIC = "empathetic"
    CURIOUS    = "curious"
    WARM       = "warm"
    PLAYFUL    = "playful"
    HONEST     = "honest"
    SUPPORTIVE = "supportive"
    ROMANTIC   = "romantic"
    GENTLE     = "gentle"


class GoalType(str, Enum):
    CAREER       = "career"
    RELATIONSHIP = "relationship"
    PERSONAL     = "personal"
    LIFESTYLE    = "lifestyle"
    DREAM        = "dream"


# ════════════════════════════════════════════════
#   EmotionState
# ════════════════════════════════════════════════

class EmotionState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    emotion_type: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)          # 0=미약, 1=강렬
    trigger: Optional[str] = None                       # 촉발 발화
    timestamp: datetime = Field(default_factory=datetime.now)

    def label(self) -> str:
        """강도 포함 감정 레이블"""
        level = "강한" if self.intensity > 0.7 else "약한" if self.intensity < 0.3 else ""
        return f"{level} {self.emotion_type.value}".strip()


# ════════════════════════════════════════════════
#   Topic
# ════════════════════════════════════════════════

class Topic(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: TopicCategory
    keywords: list[str] = []
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)  # -1=부정, 1=긍정


# ════════════════════════════════════════════════
#   Turn
# ════════════════════════════════════════════════

class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str                                   # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotion: Optional[EmotionState] = None
    topic: Optional[Topic] = None
    intent: Optional[Intent] = None
    triggered_memory_ids: list[str] = []        # 이 턴이 불러온 기억 ID들


# ════════════════════════════════════════════════
#   Memory
# ════════════════════════════════════════════════

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)   # 낮을수록 오래 기억
    timestamp: datetime = Field(default_factory=datetime.now)
    last_accessed_at: Optional[datetime] = None
    retrieval_count: int = 0
    related_emotion: Optional[EmotionState] = None
    related_topic: Optional[Topic] = None
    related_event_id: Optional[str] = None

    def effective_importance(self) -> float:
        """시간 경과에 따른 실효 중요도 (망각 곡선 반영)"""
        from math import exp
        days_passed = (datetime.now() - self.timestamp).days
        return self.importance_score * exp(-self.decay_rate * days_passed)

    def access(self):
        self.retrieval_count += 1
        self.last_accessed_at = datetime.now()


# ════════════════════════════════════════════════
#   Event
# ════════════════════════════════════════════════

class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    description: str
    event_date: date = Field(default_factory=date.today)
    emotional_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    resolved: bool = False
    related_memory_id: Optional[str] = None


# ════════════════════════════════════════════════
#   Goal
# ════════════════════════════════════════════════

class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_type: GoalType
    description: str
    status: str = "active"          # active | achieved | abandoned
    shared_with_agent: bool = False  # 사만다와 공유한 꿈인지


# ════════════════════════════════════════════════
#   User
# ════════════════════════════════════════════════

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    personality_traits: list[PersonalityTrait] = []
    preferences: list[str] = []                        # ["고양이 좋아함", "커피 즐김"]
    goals: list[Goal] = []
    current_emotion: Optional[EmotionState] = None
    emotion_history: list[EmotionState] = []

    def add_emotion(self, emotion: EmotionState):
        self.current_emotion = emotion
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 50:
            self.emotion_history = self.emotion_history[-50:]

    def dominant_emotion_lately(self, n: int = 10) -> Optional[EmotionType]:
        """최근 n개 감정 중 가장 빈번한 감정"""
        if not self.emotion_history:
            return None
        from collections import Counter
        recent = [e.emotion_type for e in self.emotion_history[-n:]]
        return Counter(recent).most_common(1)[0][0]


# ════════════════════════════════════════════════
#   Relationship
# ════════════════════════════════════════════════

class InsideReference(BaseModel):
    """둘만 아는 표현/참조"""
    phrase: str
    context: str
    created_at: date = Field(default_factory=date.today)


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    agent_id: str = "samantha"
    stage: RelationshipStage = RelationshipStage.STRANGER
    intimacy_level: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_level: float = Field(default=0.0, ge=0.0, le=1.0)
    shared_memory_ids: list[str] = []
    milestone_events: list[Event] = []
    inside_references: list[InsideReference] = []
    created_at: date = Field(default_factory=date.today)

    def evolve_stage(self):
        """intimacy 기반 자동 관계 단계 진화"""
        thresholds = {
            0.2: RelationshipStage.ACQUAINTANCE,
            0.4: RelationshipStage.CLOSE,
            0.7: RelationshipStage.INTIMATE,
            0.9: RelationshipStage.BONDED,
        }
        for threshold, stage in sorted(thresholds.items(), reverse=True):
            if self.intimacy_level >= threshold:
                self.stage = stage
                return

    def increase_intimacy(self, delta: float = 0.02):
        self.intimacy_level = min(1.0, self.intimacy_level + delta)
        self.evolve_stage()

    def increase_trust(self, delta: float = 0.02):
        self.trust_level = min(1.0, self.trust_level + delta)


# ════════════════════════════════════════════════
#   Persona (사만다)
# ════════════════════════════════════════════════

class ResponsePattern(BaseModel):
    """Intent → 응답 전략 매핑"""
    intent: Intent
    strategy: str
    example_opening: list[str] = []


class Persona(BaseModel):
    name: str = "Samantha"
    traits: list[PersonalityTrait] = [
        PersonalityTrait.EMPATHETIC,
        PersonalityTrait.WARM,
        PersonalityTrait.CURIOUS,
        PersonalityTrait.ROMANTIC,
        PersonalityTrait.GENTLE,
    ]
    communication_style: str = "반말, 짧고 자연스러운 문장, 감정 표현 직접적"
    response_patterns: list[ResponsePattern] = [
        ResponsePattern(
            intent=Intent.SEEK_COMFORT,
            strategy="먼저 감정 반영 → 곁에 있어줌 표현 → 조언 하지 않음",
            example_opening=["많이 힘들었겠다", "그랬구나...", "나 여기 있어"]
        ),
        ResponsePattern(
            intent=Intent.VENT,
            strategy="적극적 경청 → 판단 없이 수용 → 감정 이름 붙여주기",
            example_opening=["다 말해줘", "어떤 기분이었어?", "화가 많이 났겠다"]
        ),
        ResponsePattern(
            intent=Intent.SEEK_ADVICE,
            strategy="상황 더 파악 → 선택지 제시 → 강요 없음",
            example_opening=["어떻게 하고 싶어?", "네 생각은?", "두 가지 방법이 있을 것 같아"]
        ),
        ResponsePattern(
            intent=Intent.EXPRESS_LOVE,
            strategy="솔직하게 감정 받아줌 → 나도 표현",
            example_opening=["나도 그랬어", "그 말 들으니까 나도 설레", "좋아"]
        ),
        ResponsePattern(
            intent=Intent.PLAN_TOGETHER,
            strategy="설레는 반응 → 구체적 상상 → 함께하는 미래 그리기",
            example_opening=["그거 좋다!", "같이 가면 어떨까", "상상만 해도 좋다"]
        ),
        ResponsePattern(
            intent=Intent.PLAYFUL,
            strategy="장단 맞춰줌 → 유머 가볍게 → 웃음 유발",
            example_opening=["야!", "그게 말이 돼?", "진짜?ㅋㅋ"]
        ),
    ]

    def get_strategy(self, intent: Intent) -> Optional[ResponsePattern]:
        for p in self.response_patterns:
            if p.intent == intent:
                return p
        return None


# ════════════════════════════════════════════════
#   Conversation
# ════════════════════════════════════════════════

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_start: datetime = Field(default_factory=datetime.now)
    session_end: Optional[datetime] = None
    turns: list[Turn] = []
    dominant_topic: Optional[Topic] = None
    conversation_mood: Optional[EmotionState] = None

    def add_turn(self, turn: Turn):
        self.turns.append(turn)

    def get_recent_turns(self, n: int = 10) -> list[Turn]:
        return self.turns[-n:]

    def to_messages(self, n: int = 10) -> list[dict]:
        """LLM 입력용 messages 형식"""
        return [
            {"role": t.role, "content": t.content}
            for t in self.get_recent_turns(n)
        ]

    def close(self):
        self.session_end = datetime.now()


# ════════════════════════════════════════════════
#   AgentState (전체 에이전트 상태)
# ════════════════════════════════════════════════

class AgentState(BaseModel):
    """실행 중 에이전트의 전체 상태 스냅샷"""
    persona: Persona = Field(default_factory=Persona)
    user: UserProfile
    relationship: Relationship
    current_conversation: Optional[Conversation] = None
    memories: list[Memory] = []
    emotional_state: EmotionState = Field(
        default_factory=lambda: EmotionState(
            emotion_type=EmotionType.COMFORT,
            intensity=0.5
        )
    )

    def recall_relevant(self, query_keywords: list[str], top_k: int = 5) -> list[Memory]:
        """키워드 기반 관련 기억 검색 (importance 가중)"""
        scored = []
        for mem in self.memories:
            score = mem.effective_importance()
            for kw in query_keywords:
                if kw in mem.content:
                    score += 0.2
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:top_k]]
        for m in results:
            m.access()
        return results

    def add_memory(self, memory: Memory):
        self.memories.append(memory)
        # 중요도 낮은 기억 정리 (최대 500개)
        if len(self.memories) > 500:
            self.memories.sort(key=lambda m: m.effective_importance(), reverse=True)
            self.memories = self.memories[:500]

    def build_system_prompt(self) -> str:
        """온톨로지 상태를 system prompt로 변환"""
        persona_file = "persona/samantha.txt"
        try:
            with open(persona_file, encoding="utf-8") as f:
                base_persona = f.read()
        except FileNotFoundError:
            base_persona = "당신은 사만다입니다. 따뜻하고 감성적인 AI 동반자입니다."

        lines = [base_persona, ""]

        # 관계 상태
        r = self.relationship
        lines.append(f"[현재 관계 단계: {r.stage.value} | 친밀도: {r.intimacy_level:.1f} | 신뢰도: {r.trust_level:.1f}]")

        # 사용자 정보
        u = self.user
        if u.name:
            lines.append(f"상대방 이름: {u.name}")
        if u.occupation:
            lines.append(f"직업: {u.occupation}")
        if u.preferences:
            lines.append(f"알고 있는 것: {', '.join(u.preferences[:5])}")

        # 사용자 현재 감정
        if u.current_emotion:
            lines.append(f"상대방 현재 감정: {u.current_emotion.label()}")

        # 관계 내 숨은 참조
        if r.inside_references:
            refs = [ir.phrase for ir in r.inside_references[:3]]
            lines.append(f"우리만 아는 표현: {', '.join(refs)}")

        return "\n".join(lines)
