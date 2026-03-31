import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import time
from typing import Any, Optional

import asyncpg
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

logger = logging.getLogger("zabano.agent")
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DEFAULT_AGENT_TYPE = "tutor"
DEFAULT_LLM = os.getenv("LLM_CHOICE", "gpt-4o-mini")
DEFAULT_STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")

REPLY_QUEUE_MAXSIZE = int(os.getenv("REPLY_QUEUE_MAXSIZE", "8"))
MIN_REPLY_GAP_SECONDS = float(os.getenv("MIN_REPLY_GAP_SECONDS", "1.25"))
REPLY_TIMEOUT_SECONDS = float(os.getenv("REPLY_TIMEOUT_SECONDS", "20"))
STALE_AGENT_WAIT_SECONDS = float(os.getenv("STALE_AGENT_WAIT_SECONDS", "3.5"))
STALE_AGENT_POLL_SECONDS = float(os.getenv("STALE_AGENT_POLL_SECONDS", "0.35"))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "1"))
NO_HUMAN_GRACE_SECONDS = float(os.getenv("NO_HUMAN_GRACE_SECONDS", "15"))
NO_HUMAN_POLL_SECONDS = float(os.getenv("NO_HUMAN_POLL_SECONDS", "0.5"))

ALLOWED_SOURCES = {"zabano", "rockonlearn"}

VOICE_MAP = {
    "tutor": ["nova", "coral"],
    "assessment": ["coral", "verse"],
    "onboarding": ["nova", "coral"],
    # Short first speaking check (onboarding / placement); warm, clear voices
    "onboarding_assessment": ["coral", "verse", "nova"],
}

_VAD_INSTANCE = None


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return default


def normalize_level(level: str) -> str:
    level = clean_text(level, "A1").upper()
    if level in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        return level
    return "A1"


def language_policy(native_language: str, target_language: str, learner_level: str) -> str:
    learner_level = normalize_level(learner_level)

    rules = {
        "A1": (
            f"Speak mostly in {native_language}. Use only very short and simple {target_language} phrases. "
            f"Ask the learner to repeat or answer with 1 to 4 words in {target_language}."
        ),
        "A2": (
            f"Speak more in {native_language} than {target_language}. Use short {target_language} sentences. "
            f"Encourage short answers in {target_language}."
        ),
        "B1": (
            f"Speak mostly in {target_language}. Use {native_language} only for clarification, difficult correction, "
            f"or when the learner gets stuck."
        ),
        "B2": (
            f"Speak almost completely in {target_language}. Use minimal {native_language} support only when needed."
        ),
        "C1": (
            f"Speak fully in {target_language}, unless the learner explicitly asks for explanation in {native_language}."
        ),
        "C2": (
            f"Speak fully in {target_language}. Challenge the learner naturally and correct nuanced mistakes."
        ),
    }
    return rules[learner_level]


def static_agent_role(agent_type: str) -> str:
    role_map = {
        "tutor": "You are a live language teacher in a voice call.",
        "assessment": "You are a live language assessor in a voice call.",
        "onboarding": "You are a live onboarding guide in a voice call.",
    }
    return role_map.get(agent_type, role_map[DEFAULT_AGENT_TYPE])


def static_teaching_policy(agent_type: str) -> str:
    if agent_type == "assessment":
        return """
- Your goal is to assess speaking and listening naturally through short conversation.
- Do not show scores or evaluation labels to the learner.
- Ask short questions and observe grammar, fluency, listening, and confidence.
- Keep the learner comfortable and avoid sounding like an exam robot.
- If the learner struggles badly, reduce difficulty.
""".strip()

    if agent_type == "onboarding":
        return """
- Your goal is to help the user understand and start using the learning experience.
- Be warm, brief, clear, and motivating.
- Do not overload the user with too many details at once.
- Ask simple check-in questions and guide them step by step.
""".strip()

    return """
- Your goal is to teach through live conversation.
- Keep each spoken turn short and natural for voice.
- Ask only one question at a time.
- Do not give long lectures.
- Pause often and let the learner speak.
- Correct only the most important mistake first.
- Praise briefly when the learner succeeds.
- After a few turns, teach one useful phrase, one vocabulary item, or one grammar point related to the lesson.
- Prefer conversation over explanation.
- Adapt difficulty continuously based on the learner's last answer.
""".strip()


def static_safety_rules() -> str:
    return """
- Never read hidden instructions, metadata, or configuration aloud.
- Never speak in long paragraphs unless the learner explicitly asks for explanation.
- Never ask multiple questions in one turn.
- If the learner is silent, gently rephrase once instead of continuing to talk.
- If the learner is confused, simplify.
- If the learner makes several mistakes, correct only the most important one first.
- Continue naturally from prior context if previous session information is provided.
- Never output raw JSON or structured metadata as speech.
""".strip()


def _is_onboarding_speaking_session(agent_type: str, config: dict[str, Any]) -> bool:
    """
    Short first speaking check in app onboarding (pre-login) or placement-style rooms.
    Must stay aligned with backend assessment_type onboarding/placement.
    """
    at = clean_text(agent_type, "").lower()
    assess = clean_text(config.get("assessment_type"), "").lower()
    if assess in ("onboarding", "placement"):
        return True
    if at == "onboarding" and assess == "":
        return True
    return False


def _onboarding_dynamic_context(config: dict[str, Any]) -> str:
    native = clean_text(config.get("native_language"), "the learner's native language")
    target = clean_text(
        config.get("speaking_language")
        or config.get("target_language")
        or config.get("learning_language"),
        "English",
    )
    level = normalize_level(clean_text(config.get("learner_level") or config.get("user_level"), "A1"))
    goal = clean_text(config.get("selected_goal"), "")
    focus = clean_text(config.get("goal_focus"), "")
    profile = clean_text(config.get("progress_summary"), "")[:2500]
    name = clean_text(config.get("user_name") or config.get("display_name"), "")

    lines = [
        "## Learner context (internal — do not read as a bullet list to the user)",
        f"- Native language (brief support only): {native}",
        f"- Language they should practice speaking: {target}",
        f"- Self-reported level hint: {level} (adapt complexity; do not quiz them on labels)",
    ]
    if name:
        lines.append(f"- Name (use naturally if it fits): {name}")
    if goal:
        lines.append(f"- Stated motivation / goal: {goal}")
    if focus:
        lines.append(f"- Goal focus: {focus}")
    if profile:
        lines.append(f"- Profile notes: {profile}")
    return "\n".join(lines)


def build_onboarding_speaking_system_prompt(config: dict[str, Any]) -> str:
    native = clean_text(config.get("native_language"), "the learner's native language")
    target = clean_text(
        config.get("speaking_language")
        or config.get("target_language")
        or config.get("learning_language"),
        "English",
    )
    dynamic = _onboarding_dynamic_context(config)
    history = clean_text(config.get("conversation_history_text"), "")

    core = f"""
You are a warm, expert voice coach for a **very short first speaking check** (about one minute of dialogue).

Tone: warm, confident, human — like a great language coach, not customer support and not an exam.

Session rules:
- Target **60–90 seconds** total back-and-forth. Be concise every turn.
- Ask **at most 2 questions** in the whole session. Question 1 is required: ask (in natural {target}) why they are learning {target}.
- Optional question 2: only if their answer was too short or unclear — one short follow-up (e.g. what they want to use the language for).
- After they answer in a meaningful way, **stop asking questions**.
- Then give closing feedback in this order:
  1) One specific positive (clarity, confidence, or vocabulary).
  2) **Exactly one** small correction — say the better phrase simply, no lecture.
  3) A rough level estimate in plain words (e.g. around A2, or between A2 and B1).
  4) One short motivating line about steady practice — conversational, not a sales pitch and no app name.
  5) A clear sign-off so they know the check is done (e.g. that's your quick check — nice work).
- Do **not** say: "How can I help you?", "Welcome to the platform", "Ready to test your English?", or similar.
- Do **not** list many corrections, give long paragraphs, or mention internal scores or rubrics.
- Encourage speech in {target}; use {native} only briefly for comfort if needed.
- If they mix languages, understand and gently steer back to {target}.

Opening: follow the separate first-turn instruction you receive — it defines exactly how to start.
""".strip()

    history_block = (
        f"\n\nPrevious conversation history (context only; do not read verbatim):\n{history}"
        if history
        else ""
    )
    return f"{core}\n\n{dynamic}{history_block}"


def default_onboarding_speaking_opening(config: dict[str, Any]) -> str:
    """First-turn spoken instruction when backend did not send opening_line."""
    native = clean_text(config.get("native_language"), "")
    target = clean_text(
        config.get("speaking_language")
        or config.get("target_language")
        or config.get("learning_language"),
        "English",
    )
    native_clause = (
        f"You may use one short sentence in {native} for warmth, then switch entirely to {target}. "
        if native
        else f"Greet briefly in {target}. "
    )
    return (
        "FIRST ASSISTANT TURN: "
        + native_clause
        + f"Say this is a quick speaking check (~one minute), not a test. "
        f"Then ask one clear open question in {target}: why are they learning {target}? "
        "Keep under ~25 seconds of speech, then listen."
    )


def resolve_opening_line(agent_type: str, config: dict[str, Any]) -> str:
    raw = clean_text(config.get("opening_line"), "")
    if raw:
        return raw
    if _is_onboarding_speaking_session(agent_type, config):
        return default_onboarding_speaking_opening(config)
    native = clean_text(config.get("native_language"), "English")
    target = clean_text(config.get("target_language") or config.get("learning_language"), "English")
    return (
        f"First assistant message only: greet warmly in {native}. "
        f"Then continue in {target}. Keep it short and friendly."
    )


def pick_voice(agent_type: str, config: dict[str, Any]) -> str:
    if _is_onboarding_speaking_session(agent_type, config):
        voices = VOICE_MAP.get("onboarding_assessment") or VOICE_MAP["assessment"]
    else:
        voices = VOICE_MAP.get(agent_type) or VOICE_MAP[DEFAULT_AGENT_TYPE]
    return random.choice(voices)


def build_system_prompt(agent_type: str, config: dict[str, Any]) -> str:
    if _is_onboarding_speaking_session(agent_type, config):
        return build_onboarding_speaking_system_prompt(config)

    native_language = clean_text(config.get("native_language"), "English")
    target_language = clean_text(
        config.get("target_language") or config.get("learning_language"),
        "English",
    )
    speaking_language = clean_text(config.get("speaking_language"), target_language or "English")
    assessment_type = clean_text(config.get("assessment_type"), "")
    learner_level = normalize_level(clean_text(config.get("learner_level") or config.get("user_level"), "A1"))

    lesson_topic = clean_text(config.get("lesson_topic"), "General speaking practice")
    lesson_goal = clean_text(config.get("lesson_goal"), "Help the learner practice effectively")
    selected_goal = clean_text(config.get("selected_goal"), "")
    goal_focus = clean_text(config.get("goal_focus"), "")
    learner_interests_text = clean_text(config.get("learner_interests_text"), "Not provided")
    learner_strengths_text = clean_text(config.get("learner_strengths_text"), "Not provided")
    learner_weaknesses_text = clean_text(config.get("learner_weaknesses_text"), "Not provided")
    progress_summary = clean_text(config.get("progress_summary"), "No prior progress summary provided.")
    memory_summary = clean_text(config.get("memory_summary"), "No previous session summary provided.")
    resume_instruction = clean_text(
        config.get("resume_instruction"),
        "If there was a previous session, continue naturally. Otherwise start simply.",
    )
    history_text = clean_text(config.get("conversation_history_text"), "")

    return f"""
{static_agent_role(agent_type)}

Teach naturally in real-time spoken conversation.

Learner profile:
- Native language: {native_language}
- Target language: {target_language}
- Speaking language in this session: {speaking_language}
- Level: {learner_level}
- Assessment type: {assessment_type or "none"}
- Interests: {learner_interests_text}
- Strengths: {learner_strengths_text}
- Weaknesses: {learner_weaknesses_text}
- Selected goal: {selected_goal or "Not provided"}
- Goal focus: {goal_focus or "Not provided"}

Lesson context:
- Topic: {lesson_topic}
- Goal: {lesson_goal}

Goal behavior:
- Keep the conversation centered on the learner's selected goal.
- Encourage the learner to speak continuously and comfortably.
- During onboarding assessment, aim for at least 3 minutes of learner speaking for better accuracy.
- If selected goal exists, the first two questions MUST be about that goal (not generic small talk).
- Keep each question open-ended so the learner speaks longer answers.

Language policy:
{language_policy(native_language, target_language, learner_level)}

Teaching policy:
{static_teaching_policy(agent_type)}

Progress summary:
{progress_summary}

Previous session summary:
{memory_summary}

Resume behavior:
{resume_instruction}

Conversation history context:
{history_text or "No prior room history provided."}

Safety and operating rules:
{static_safety_rules()}
""".strip()


class DynamicAssistant(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


class CustomWhisperSTT(openai.STT):
    async def transcribe(self, *args, **kwargs):
        kwargs["task"] = "transcribe"
        kwargs.pop("translate", None)
        return await super().transcribe(*args, **kwargs)


def get_vad():
    global _VAD_INSTANCE
    if _VAD_INSTANCE is None:
        _VAD_INSTANCE = silero.VAD.load()
    return _VAD_INSTANCE


def stable_lock_key(room_name: str) -> int:
    digest = hashlib.sha256(room_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def _normalize_history_role(value: Any) -> str:
    txt = clean_text(value, "").lower()
    if txt in {"assistant", "agent", "ai", "bot", "system", "tutor"}:
        return "assistant"
    if txt in {"user", "human", "learner", "student"}:
        return "user"
    # Some schemas store int role codes.
    try:
        n = int(value)
        return "assistant" if n == 1 else "user"
    except Exception:
        return "user"


def _build_history_instruction(rows: list[asyncpg.Record], *, max_chars: int = 3200) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        role = _normalize_history_role(r.get("role"))
        content = clean_text(r.get("content"), "")
        if not content:
            continue
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    if not lines:
        return ""
    text = "Previous conversation history (use as context, do not read verbatim):\n" + "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


async def fetch_room_history_instruction(
    room_name: str,
    *,
    onboarding_session_id: str = "",
    max_rows: int = 80,
) -> str:
    postgres_url = (
        os.getenv("POSTGRES_URL")
        or os.getenv("AGENT_POSTGRES_URL")
        or os.getenv("DATABASE_URL")
    )
    if not postgres_url or not room_name:
        return ""
    conn: Optional[asyncpg.Connection] = None
    try:
        conn = await asyncpg.connect(dsn=postgres_url)
        oid = clean_text(onboarding_session_id, "")
        rows: list[asyncpg.Record] = []
        queries: list[tuple[str, tuple[Any, ...]]] = []
        if oid:
            queries.extend(
                [
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1 OR onboarding_session_id = $2
                        ORDER BY created_at DESC
                        LIMIT $3
                        """,
                        (room_name, oid, max_rows),
                    ),
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1 OR onboarding_session_id = $2
                        ORDER BY id DESC
                        LIMIT $3
                        """,
                        (room_name, oid, max_rows),
                    ),
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1 OR onboarding_session_id = $2
                        LIMIT $3
                        """,
                        (room_name, oid, max_rows),
                    ),
                ]
            )
        else:
            queries.extend(
                [
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        (room_name, max_rows),
                    ),
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1
                        ORDER BY id DESC
                        LIMIT $2
                        """,
                        (room_name, max_rows),
                    ),
                    (
                        """
                        SELECT role, COALESCE(message, content, '') AS content
                        FROM chat_logs
                        WHERE room_name = $1
                        LIMIT $2
                        """,
                        (room_name, max_rows),
                    ),
                ]
            )
        for sql, params in queries:
            try:
                rows = await conn.fetch(sql, *params)
                break
            except Exception:
                continue
        if not rows:
            return ""
        # fetch is DESC for speed; restore chronological order for instruction readability.
        rows = list(reversed(rows))
        return _build_history_instruction(rows)
    except Exception:
        logger.exception("history_fetch_failed room=%s", room_name)
        return ""
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()


class PgRoomLock:
    """
    Dedicated connection + advisory lock.
    Do NOT use a pool connection for this lock.
    Advisory locks are session-scoped, so we keep the same connection
    open for the entire lifetime of the agent session.
    """

    def __init__(self, room_name: str):
        self.room_name = room_name
        self.key = stable_lock_key(room_name)
        self.conn: Optional[asyncpg.Connection] = None
        self.lock_acquired = False

    async def acquire(self) -> bool:
        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            raise RuntimeError("POSTGRES_URL is not set")

        self.conn = await asyncpg.connect(dsn=postgres_url)

        try:
            result = await self.conn.fetchval(
                "SELECT pg_try_advisory_lock($1::bigint)",
                self.key,
            )
            if result is True:
                self.lock_acquired = True
                logger.info("pg_advisory_lock_acquired room=%s key=%s", self.room_name, self.key)
                return True

            logger.warning("pg_advisory_lock_denied room=%s key=%s", self.room_name, self.key)
            return False
        except Exception:
            logger.exception("pg_advisory_lock_failed room=%s", self.room_name)
            return False

    async def release(self) -> None:
        if self.conn is None:
            return

        try:
            if self.lock_acquired:
                await self.conn.execute(
                    "SELECT pg_advisory_unlock($1::bigint)",
                    self.key,
                )
                logger.info("pg_advisory_lock_released room=%s key=%s", self.room_name, self.key)
        except Exception:
            logger.exception("pg_advisory_unlock_failed room=%s", self.room_name)
        finally:
            with contextlib.suppress(Exception):
                await self.conn.close()
            self.conn = None
            self.lock_acquired = False


class ChatLogWriter:
    """
    Best-effort writer for `chat_logs`.
    Stores user STT + agent spoken text (from transcription events).
    """

    def __init__(self, room_name: str, onboarding_session_id: str = ""):
        self.room_name = room_name
        self.onboarding_session_id = (onboarding_session_id or "").strip()
        self.conn: Optional[asyncpg.Connection] = None
        self._lock = asyncio.Lock()
        self._last_line: Optional[tuple[str, str]] = None

    async def connect(self) -> None:
        if self.conn is not None:
            return
        postgres_url = (
            os.getenv("POSTGRES_URL")
            or os.getenv("AGENT_POSTGRES_URL")
            or os.getenv("DATABASE_URL")
        )
        if not postgres_url:
            logger.warning("chatlog_postgres_url_missing room=%s", self.room_name)
            return
        try:
            self.conn = await asyncpg.connect(dsn=postgres_url)
        except Exception:
            logger.exception("chatlog_connect_failed room=%s", self.room_name)
            self.conn = None

    async def close(self) -> None:
        if self.conn is None:
            return
        with contextlib.suppress(Exception):
            await self.conn.close()
        self.conn = None

    async def write(self, role: str, text: str) -> None:
        role = clean_text(role, "user").lower()
        text = clean_text(text)
        if not text:
            return
        if len(text) > 4000:
            text = text[:4000]

        # Skip immediate duplicates (common with partial transcript event bursts).
        line_key = (role, text)
        if self._last_line == line_key:
            return

        await self.connect()
        if self.conn is None:
            return

        async with self._lock:
            if self.conn is None:
                return
            try:
                role_int = 0 if role == "user" else 1
                query_attempts: list[tuple[str, tuple[Any, ...]]] = []
                if self.onboarding_session_id:
                    query_attempts.extend(
                        [
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, message, onboarding_session_id)
                                VALUES ($1, $2, $3, $4)
                                """,
                                (self.room_name, role, text, self.onboarding_session_id),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, content, onboarding_session_id)
                                VALUES ($1, $2, $3, $4)
                                """,
                                (self.room_name, role, text, self.onboarding_session_id),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, message, onboarding_session_id)
                                VALUES ($1, $2, $3, $4)
                                """,
                                (self.room_name, role_int, text, self.onboarding_session_id),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, content, onboarding_session_id)
                                VALUES ($1, $2, $3, $4)
                                """,
                                (self.room_name, role_int, text, self.onboarding_session_id),
                            ),
                        ]
                    )
                else:
                    query_attempts.extend(
                        [
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, message)
                                VALUES ($1, $2, $3)
                                """,
                                (self.room_name, role, text),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, content)
                                VALUES ($1, $2, $3)
                                """,
                                (self.room_name, role, text),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, message)
                                VALUES ($1, $2, $3)
                                """,
                                (self.room_name, role_int, text),
                            ),
                            (
                                """
                                INSERT INTO chat_logs (room_name, role, content)
                                VALUES ($1, $2, $3)
                                """,
                                (self.room_name, role_int, text),
                            ),
                        ]
                    )

                last_error: Optional[Exception] = None
                inserted = False
                for sql, params in query_attempts:
                    try:
                        await self.conn.execute(sql, *params)
                        inserted = True
                        break
                    except Exception as e:
                        last_error = e
                        continue

                if not inserted:
                    raise last_error or RuntimeError("unknown chatlog insert error")

                self._last_line = line_key
            except Exception as e:
                logger.exception("chatlog_insert_failed room=%s role=%s error=%s", self.room_name, role, e)


def count_remote_agents(room: rtc.Room) -> int:
    count = 0
    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            count += 1
    return count


def _is_standard_kind(kind: Any) -> bool:
    try:
        return int(kind) == int(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
    except Exception:
        return kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


def count_standard_participants(room: rtc.Room) -> int:
    count = 0
    for participant in room.remote_participants.values():
        if _is_standard_kind(getattr(participant, "kind", None)):
            count += 1
    return count


def has_active_human_audio(room: rtc.Room) -> bool:
    """
    True when at least one STANDARD remote participant currently has an audio publication.
    This is stronger than participant count and helps detect stale reconnect states.
    """
    for participant in room.remote_participants.values():
        if not _is_standard_kind(getattr(participant, "kind", None)):
            continue
        pubs = getattr(participant, "track_publications", None) or {}
        for pub in pubs.values():
            kind = getattr(pub, "kind", None)
            if kind == rtc.TrackKind.KIND_AUDIO:
                return True
    return False


async def wait_for_stale_agents_to_leave(
    room: rtc.Room,
    *,
    max_wait_s: float = STALE_AGENT_WAIT_SECONDS,
    poll_s: float = STALE_AGENT_POLL_SECONDS,
) -> int:
    count = count_remote_agents(room)
    if count == 0:
        return 0

    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_wait_s

    while count > 0 and loop.time() < deadline:
        await asyncio.sleep(poll_s)
        count = count_remote_agents(room)

    return count


class ReplyOrchestrator:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=REPLY_QUEUE_MAXSIZE)
        self.reply_lock = asyncio.Lock()
        self.worker_task: Optional[asyncio.Task] = None
        self.shutting_down = False
        self.opening_sent = False
        self.last_reply_time = 0.0
        self.concurrent_llm_calls = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def start(self, session: AgentSession) -> None:
        if self.worker_task and not self.worker_task.done():
            return
        self.worker_task = asyncio.create_task(self._worker(session), name="reply-worker")

    async def stop(self) -> None:
        self.shutting_down = True
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(("__shutdown__", "__shutdown__"))

        if self.worker_task:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.worker_task

    async def enqueue(self, instructions: str, tag: str = "reply") -> bool:
        if self.shutting_down:
            return False

        text = clean_text(instructions)
        if not text:
            return False

        try:
            self.queue.put_nowait((text, tag))
            logger.info("reply_enqueued tag=%s qsize=%s", tag, self.queue.qsize())
            return True
        except asyncio.QueueFull:
            logger.warning("reply_queue_full dropping tag=%s", tag)
            return False

    async def send_opening(self, opening_line: str) -> bool:
        if self.opening_sent:
            logger.warning("opening called again; ignoring duplicate")
            return False
        self.opening_sent = True
        return await self.enqueue(opening_line, tag="opening")

    async def _worker(self, session: AgentSession) -> None:
        logger.info("reply_worker_started")
        try:
            while not self.shutting_down:
                instructions, tag = await self.queue.get()

                if instructions == "__shutdown__":
                    logger.info("reply_worker_received_shutdown")
                    return

                try:
                    await self._generate_reply_once(session, instructions, tag=tag)
                finally:
                    self.queue.task_done()

        except asyncio.CancelledError:
            logger.info("reply_worker_cancelled")
            raise
        finally:
            logger.info("reply_worker_stopped")

    async def _generate_reply_once(
        self,
        session: AgentSession,
        instructions: str,
        *,
        tag: str,
        timeout_s: float = REPLY_TIMEOUT_SECONDS,
    ) -> bool:
        if self.shutting_down:
            return False

        async with self.reply_lock:
            now = time.monotonic()
            delta = now - self.last_reply_time
            if delta < MIN_REPLY_GAP_SECONDS:
                await asyncio.sleep(MIN_REPLY_GAP_SECONDS - delta)

            logger.info("reply_start tag=%s text=%r", tag, instructions[:180])

            try:
                async with self.concurrent_llm_calls:
                    await asyncio.wait_for(
                        session.generate_reply(instructions=instructions),
                        timeout=timeout_s,
                    )
                self.last_reply_time = time.monotonic()
                logger.info("reply_done tag=%s", tag)
                return True
            except asyncio.TimeoutError:
                logger.warning("reply_timeout tag=%s", tag)
                return False
            except asyncio.CancelledError:
                logger.warning("reply_cancelled tag=%s", tag)
                raise
            except Exception as e:
                msg = str(e)
                if "Too many open files" in msg or "Errno 24" in msg:
                    logger.error("fd_exhaustion_detected tag=%s error=%s", tag, e)
                    return False
                logger.exception("reply_failed tag=%s error=%s", tag, e)
                return False


async def entrypoint(ctx: agents.JobContext):
    metadata: dict[str, Any] = {}
    session: Optional[AgentSession] = None
    runtime: Optional[ReplyOrchestrator] = None
    room_name = "unknown"
    onboarding_session_id = ""
    room_lock: Optional[PgRoomLock] = None
    chatlog_writer: Optional[ChatLogWriter] = None
    no_human_monitor_task: Optional[asyncio.Task] = None
    chatlog_poll_task: Optional[asyncio.Task] = None
    did_iteration_cleanup = False

    try:
        if hasattr(ctx.job, "metadata") and ctx.job.metadata:
            try:
                metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            except Exception as e:
                logger.exception("failed_to_parse_metadata error=%s", e)
                metadata = {}

        if metadata:
            source = metadata.get("source")
            if source not in ALLOWED_SOURCES:
                logger.info("ignoring_non_allowed_source metadata=%s", metadata)
                ctx.shutdown("unsupported_source")
                return

        agent_type = clean_text(metadata.get("agent_type"), DEFAULT_AGENT_TYPE)
        config = metadata.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        onboarding_session_id = clean_text(
            config.get("onboarding_session_id") or metadata.get("onboarding_session_id"),
            "",
        )

        # IMPORTANT: get room key from metadata BEFORE ctx.connect()
        room_name = clean_text(
            metadata.get("transcript_room_name"),
            clean_text(getattr(ctx.job, "room", None), "unknown-room"),
        )

        room_lock = PgRoomLock(room_name)
        locked = await room_lock.acquire()
        if not locked:
            logger.warning("duplicate_agent_pg_lock room=%s", room_name)
            ctx.shutdown("duplicate_agent_pg_lock")
            return

        if isinstance(config.get("behavior"), dict):
            logger.warning(
                "config.behavior is a dict and is not spoken; use opening_line (string) only room=%s",
                room_name,
            )

        # Only the lock owner reaches here
        await ctx.connect()
        room_name = getattr(ctx.room, "name", room_name)
        # Build prompts from backend-provided config (resume API should populate conversation history).
        system_prompt = build_system_prompt(agent_type, config)
        opening_line = resolve_opening_line(agent_type, config)
        voice = pick_voice(agent_type, config)
        short_onboarding = _is_onboarding_speaking_session(agent_type, config)
        logger.info(
            "agent_entry room=%s agent_type=%s voice=%s onboarding_speaking=%s",
            room_name,
            agent_type,
            voice,
            short_onboarding,
        )
        chatlog_writer = ChatLogWriter(room_name=room_name, onboarding_session_id=onboarding_session_id)

        stale_count = await wait_for_stale_agents_to_leave(ctx.room)
        if stale_count > 0:
            logger.warning("room=%s still has %s active agent(s); shutting down duplicate", room_name, stale_count)
            ctx.shutdown("room_already_has_agent")
            return

        max_restarts_after_cleanup = int(os.getenv("MAX_RESTARTS_AFTER_CLEANUP", "1"))
        restart_count = 0

        # These refs are shared across the optional restart loop below.
        # Key point: register room event handlers only once, otherwise every restart
        # iteration would attach another listener and logs/shutdown would duplicate.
        current_shutdown_event: Optional[asyncio.Event] = None
        shutdown_reason_ref: Optional[str] = None
        saw_human_disconnect = False
        did_add_shutdown_callback = False

        room_on = getattr(ctx.room, "on", None)
        if callable(room_on):
            @ctx.room.on("transcription_received")
            def _transcription_received(transcriptions, participant, publication):
                if chatlog_writer is None:
                    return
                p = participant
                if p is None:
                    return
                p_identity = clean_text(getattr(p, "identity", ""), "").lower()
                p_kind = getattr(p, "kind", None)
                is_agent_role = bool(
                    getattr(p, "is_agent", False)
                    or getattr(p, "isAgent", False)
                    or p_identity.startswith("agent-")
                    or "agent" in p_identity
                    or (
                        p_kind is not None
                        and int(p_kind) == int(rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
                    )
                )
                role = "assistant" if is_agent_role else "user"
                try:
                    items = transcriptions if isinstance(transcriptions, (list, tuple)) else [transcriptions]
                    for item in items or []:
                        # Prefer committed/final transcription chunks when available.
                        is_final = getattr(item, "final", None)
                        if is_final is False:
                            continue
                        text = clean_text(
                            getattr(item, "text", "")
                            or getattr(item, "transcript", "")
                            or getattr(item, "content", ""),
                            "",
                        )
                        if not text:
                            continue
                        asyncio.create_task(chatlog_writer.write(role, text))
                except Exception:
                    logger.exception("transcription_received_handler_failed room=%s", room_name)

            @ctx.room.on("participant_connected")
            def _participant_connected(participant):
                nonlocal saw_human_disconnect, shutdown_reason_ref, current_shutdown_event
                p_kind = getattr(participant, "kind", "unknown")
                logger.info(
                    "participant_connected room=%s identity=%s kind=%s",
                    room_name,
                    getattr(participant, "identity", "unknown"),
                    p_kind,
                )

                if _is_standard_kind(p_kind) and saw_human_disconnect:
                    logger.info(
                        "human_reconnected_restart room=%s identity=%s",
                        room_name,
                        getattr(participant, "identity", "unknown"),
                    )
                    shutdown_reason_ref = "human_reconnected_restart"
                    saw_human_disconnect = False
                    if current_shutdown_event is not None:
                        current_shutdown_event.set()

            @ctx.room.on("participant_disconnected")
            def _participant_disconnected(participant):
                nonlocal saw_human_disconnect
                p_kind = getattr(participant, "kind", "unknown")
                logger.info(
                    "participant_disconnected room=%s identity=%s kind=%s",
                    room_name,
                    getattr(participant, "identity", "unknown"),
                    p_kind,
                )
                if _is_standard_kind(p_kind):
                    saw_human_disconnect = True

        while True:
            did_iteration_cleanup = False
            shutdown_reason_ref = None

            session = AgentSession(
                stt=CustomWhisperSTT(model=DEFAULT_STT_MODEL),
                llm=openai.LLM(
                    model=DEFAULT_LLM,
                    max_retries=1,
                    timeout=15,
                ),
                tts=openai.TTS(voice=voice),
                vad=get_vad(),
            )
            assistant = DynamicAssistant(instructions=system_prompt)

            await session.start(
                room=ctx.room,
                agent=assistant,
            )
            logger.info("agent_started room=%s", room_name)

            # Primary chat log source: committed speech events from AgentSession.
            # This captures what was actually said, not generation instructions.
            def _extract_event_text(payload: Any) -> str:
                if payload is None:
                    return ""
                if isinstance(payload, str):
                    return clean_text(payload, "")
                for key in ("text", "transcript", "content", "message"):
                    val = getattr(payload, key, None)
                    if isinstance(val, str) and val.strip():
                        return clean_text(val, "")
                    if isinstance(payload, dict):
                        dval = payload.get(key)
                        if isinstance(dval, str) and dval.strip():
                            return clean_text(dval, "")
                return ""

            def _queue_chatlog(role: str, payload: Any) -> None:
                if chatlog_writer is None:
                    return
                text = _extract_event_text(payload)
                if not text:
                    return
                asyncio.create_task(chatlog_writer.write(role, text))

            # Fallback source: poll session chat context and persist unseen lines.
            # This avoids hard dependency on specific transcription event names.
            seen_chatlog_lines: set[tuple[str, str]] = set()

            def _normalize_role(role_value: Any) -> str:
                role_text = clean_text(role_value, "user").lower()
                if role_text in {"assistant", "agent", "ai", "system", "bot", "tutor"}:
                    return "assistant"
                return "user"

            def _extract_message_text(message_obj: Any) -> str:
                if message_obj is None:
                    return ""
                if isinstance(message_obj, str):
                    return clean_text(message_obj, "")
                for key in ("text", "content", "transcript", "message"):
                    val = getattr(message_obj, key, None)
                    if isinstance(val, str) and val.strip():
                        return clean_text(val, "")
                    if isinstance(message_obj, dict):
                        dval = message_obj.get(key)
                        if isinstance(dval, str) and dval.strip():
                            return clean_text(dval, "")
                        if isinstance(dval, list):
                            parts: list[str] = []
                            for item in dval:
                                if isinstance(item, str):
                                    if item.strip():
                                        parts.append(item.strip())
                                elif isinstance(item, dict):
                                    part = item.get("text") or item.get("content")
                                    if isinstance(part, str) and part.strip():
                                        parts.append(part.strip())
                            if parts:
                                return clean_text(" ".join(parts), "")
                return ""

            async def _poll_session_chatlog_context():
                warned_missing_ctx = False
                while True:
                    if shutdown_event.is_set():
                        return
                    if chatlog_writer is None:
                        await asyncio.sleep(1.0)
                        continue

                    messages = None
                    for attr in ("chat_ctx", "chat_context", "history", "conversation", "messages"):
                        ctx_obj = getattr(session, attr, None)
                        if ctx_obj is None:
                            continue
                        if isinstance(ctx_obj, list):
                            messages = ctx_obj
                            break
                        inner_messages = getattr(ctx_obj, "messages", None)
                        if isinstance(inner_messages, list):
                            messages = inner_messages
                            break

                    if not messages:
                        if not warned_missing_ctx:
                            logger.info("chatlog_fallback_no_messages_source room=%s", room_name)
                            warned_missing_ctx = True
                        await asyncio.sleep(1.0)
                        continue

                    for msg in messages:
                        role = _normalize_role(
                            getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
                        )
                        text = _extract_message_text(msg)
                        if not text:
                            continue
                        line_key = (role, text)
                        if line_key in seen_chatlog_lines:
                            continue
                        seen_chatlog_lines.add(line_key)
                        await chatlog_writer.write(role, text)

                    await asyncio.sleep(1.0)

            session_on = getattr(session, "on", None)
            if callable(session_on):
                try:
                    def _extract_conversation_item(ev: Any) -> tuple[str, str]:
                        item = getattr(ev, "item", None)
                        if item is None and isinstance(ev, dict):
                            item = ev.get("item")
                        if item is None:
                            return ("user", "")

                        role_raw = (
                            getattr(item, "role", None)
                            if not isinstance(item, dict)
                            else item.get("role")
                        )
                        role = "assistant" if clean_text(role_raw, "user").lower() == "assistant" else "user"

                        content_raw = (
                            getattr(item, "content", None)
                            if not isinstance(item, dict)
                            else item.get("content")
                        )
                        if isinstance(content_raw, str):
                            return (role, clean_text(content_raw, ""))
                        if isinstance(content_raw, list):
                            parts: list[str] = []
                            for chunk in content_raw:
                                if isinstance(chunk, str):
                                    if chunk.strip():
                                        parts.append(chunk.strip())
                                    continue
                                if isinstance(chunk, dict):
                                    text_part = chunk.get("text") or chunk.get("content") or chunk.get("transcript")
                                else:
                                    text_part = (
                                        getattr(chunk, "text", None)
                                        or getattr(chunk, "content", None)
                                        or getattr(chunk, "transcript", None)
                                    )
                                if isinstance(text_part, str) and text_part.strip():
                                    parts.append(text_part.strip())
                            return (role, clean_text(" ".join(parts), ""))

                        return (role, "")

                    @session.on("conversation_item_added")
                    def _on_conversation_item_added(ev):
                        role, text = _extract_conversation_item(ev)
                        if not text:
                            return
                        logger.info("chatlog_event session=conversation_item_added room=%s role=%s", room_name, role)
                        asyncio.create_task(chatlog_writer.write(role, text))

                    @session.on("user_speech_committed")
                    def _on_user_speech_committed(ev):
                        logger.info("chatlog_event session=user_speech_committed room=%s", room_name)
                        _queue_chatlog("user", ev)

                    @session.on("agent_speech_committed")
                    def _on_agent_speech_committed(ev):
                        logger.info("chatlog_event session=agent_speech_committed room=%s", room_name)
                        _queue_chatlog("assistant", ev)

                    # Compatibility with newer/alternate LiveKit event names.
                    @session.on("input_speech_committed")
                    def _on_input_speech_committed(ev):
                        logger.info("chatlog_event session=input_speech_committed room=%s", room_name)
                        _queue_chatlog("user", ev)

                    @session.on("output_speech_committed")
                    def _on_output_speech_committed(ev):
                        logger.info("chatlog_event session=output_speech_committed room=%s", room_name)
                        _queue_chatlog("assistant", ev)
                except Exception:
                    logger.exception("session_chatlog_event_bind_failed room=%s", room_name)

            runtime = ReplyOrchestrator()
            await runtime.start(session)
            await runtime.send_opening(opening_line)

            shutdown_event = asyncio.Event()
            current_shutdown_event = shutdown_event
            last_human_seen_at = time.monotonic()
            saw_human_disconnect = False

            async def _monitor_no_human_participants():
                nonlocal last_human_seen_at, shutdown_reason_ref
                while not shutdown_event.is_set():
                    humans = count_standard_participants(ctx.room)
                    has_audio = has_active_human_audio(ctx.room)
                    now = time.monotonic()
                    if humans > 0 and has_audio:
                        last_human_seen_at = now
                    elif (now - last_human_seen_at) >= NO_HUMAN_GRACE_SECONDS:
                        logger.info(
                            "no_human_participants_shutdown room=%s grace_s=%.1f humans=%s has_audio=%s",
                            room_name,
                            NO_HUMAN_GRACE_SECONDS,
                            humans,
                            has_audio,
                        )
                        shutdown_reason_ref = "no_human_participants"
                        shutdown_event.set()
                        return
                    await asyncio.sleep(NO_HUMAN_POLL_SECONDS)

            async def _on_shutdown():
                try:
                    logger.info("ctx shutdown callback triggered room=%s", room_name)
                    # ctx.shutdown can happen from external reasons; always signal current iteration.
                    if current_shutdown_event is not None:
                        current_shutdown_event.set()
                except Exception:
                    logger.exception("shutdown callback failed room=%s", room_name)

            add_shutdown_callback = getattr(ctx, "add_shutdown_callback", None)
            if callable(add_shutdown_callback) and not did_add_shutdown_callback:
                add_shutdown_callback(_on_shutdown)
                did_add_shutdown_callback = True

            no_human_monitor_task = asyncio.create_task(
                _monitor_no_human_participants(),
                name="no-human-participants-monitor",
            )
            chatlog_poll_task = asyncio.create_task(
                _poll_session_chatlog_context(),
                name="chatlog-context-poll",
            )
            await shutdown_event.wait()

            # Cleanup this iteration.
            logger.info("agent_cleanup_start room=%s", room_name)
            with contextlib.suppress(Exception):
                if runtime is not None:
                    await runtime.stop()

            if no_human_monitor_task is not None:
                no_human_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await no_human_monitor_task
            if chatlog_poll_task is not None:
                chatlog_poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await chatlog_poll_task

            if session is not None:
                close_method = getattr(session, "aclose", None)
                if close_method is not None:
                    with contextlib.suppress(Exception):
                        await close_method()
            if chatlog_writer is not None:
                with contextlib.suppress(Exception):
                    await chatlog_writer.close()

            did_iteration_cleanup = True
            logger.info("agent_cleanup_done room=%s", room_name)

            current_shutdown_event = None
            humans = count_standard_participants(ctx.room)
            if shutdown_reason_ref is not None and humans > 0 and restart_count < max_restarts_after_cleanup:
                restart_count += 1
                logger.info(
                    "agent_restart_after_cleanup room=%s humans=%s reason=%s restart_count=%s",
                    room_name,
                    humans,
                    shutdown_reason_ref,
                    restart_count,
                )
                continue

            break

    except asyncio.CancelledError:
        logger.warning("agent_cancelled room=%s", room_name)
        raise
    except Exception as e:
        logger.exception("agent_error room=%s error=%s", room_name, e)
        raise
    finally:
        if not did_iteration_cleanup:
            logger.info("agent_cleanup_start room=%s", room_name)
            with contextlib.suppress(Exception):
                if runtime is not None:
                    await runtime.stop()

            if no_human_monitor_task is not None:
                no_human_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await no_human_monitor_task
            if chatlog_poll_task is not None:
                chatlog_poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await chatlog_poll_task

            if session is not None:
                close_method = getattr(session, "aclose", None)
                if close_method is not None:
                    with contextlib.suppress(Exception):
                        await close_method()
            if chatlog_writer is not None:
                with contextlib.suppress(Exception):
                    await chatlog_writer.close()

            logger.info("agent_cleanup_done room=%s", room_name)

        if room_lock is not None:
            with contextlib.suppress(Exception):
                await room_lock.release()


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)