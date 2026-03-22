import asyncio
import contextlib
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

ALLOWED_SOURCES = {"zabano", "rockonlearn"}

VOICE_MAP = {
    "tutor": ["nova", "coral"],
    "assessment": ["coral", "verse"],
    "onboarding": ["nova"],
}

_VAD_INSTANCE = None
_DB_POOL: Optional[asyncpg.Pool] = None
_DB_POOL_LOCK = asyncio.Lock()


async def get_db_pool() -> asyncpg.Pool:
    global _DB_POOL
    if _DB_POOL is None:
        async with _DB_POOL_LOCK:
            if _DB_POOL is None:
                postgres_url = os.getenv("POSTGRES_URL")
                if not postgres_url:
                    raise RuntimeError("POSTGRES_URL is not set")
                _DB_POOL = await asyncpg.create_pool(
                    dsn=postgres_url,
                    min_size=1,
                    max_size=10,
                )
    return _DB_POOL


class PgRoomLock:
    """
    Cross-process lock using PostgreSQL advisory locks.
    This is atomic and prevents two workers from owning the same room.
    """

    def __init__(self, room_name: str):
        self.room_name = room_name
        self.conn: Optional[asyncpg.Connection] = None
        self.lock_acquired = False

    def _lock_key(self) -> int:
        # Stable enough integer key for pg advisory lock
        return abs(hash(self.room_name)) % (2**31)

    async def acquire(self) -> bool:
        pool = await get_db_pool()
        self.conn = await pool.acquire()

        try:
            result = await self.conn.fetchval(
                "SELECT pg_try_advisory_lock($1)",
                self._lock_key(),
            )
            if result:
                self.lock_acquired = True
                logger.info("pg_advisory_lock_acquired room=%s key=%s", self.room_name, self._lock_key())
                return True

            logger.warning("pg_advisory_lock_denied room=%s key=%s", self.room_name, self._lock_key())
            return False

        except Exception:
            logger.exception("pg_advisory_lock_failed room=%s", self.room_name)
            return False

    async def release(self) -> None:
        if not self.conn:
            return

        try:
            if self.lock_acquired:
                await self.conn.execute(
                    "SELECT pg_advisory_unlock($1)",
                    self._lock_key(),
                )
                logger.info("pg_advisory_lock_released room=%s key=%s", self.room_name, self._lock_key())
        except Exception:
            logger.exception("pg_advisory_unlock_failed room=%s", self.room_name)
        finally:
            try:
                pool = await get_db_pool()
                await pool.release(self.conn)
            except Exception:
                logger.exception("pg_lock_connection_release_failed room=%s", self.room_name)
            self.conn = None
            self.lock_acquired = False


def clean_text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
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


def build_system_prompt(agent_type: str, config: dict[str, Any]) -> str:
    native_language = clean_text(config.get("native_language"), "English")
    target_language = clean_text(config.get("target_language"), "English")
    learner_level = normalize_level(clean_text(config.get("learner_level"), "A1"))

    lesson_topic = clean_text(config.get("lesson_topic"), "General speaking practice")
    lesson_goal = clean_text(config.get("lesson_goal"), "Help the learner practice effectively")
    learner_interests_text = clean_text(config.get("learner_interests_text"), "Not provided")
    learner_strengths_text = clean_text(config.get("learner_strengths_text"), "Not provided")
    learner_weaknesses_text = clean_text(config.get("learner_weaknesses_text"), "Not provided")
    progress_summary = clean_text(config.get("progress_summary"), "No prior progress summary provided.")
    memory_summary = clean_text(config.get("memory_summary"), "No previous session summary provided.")
    resume_instruction = clean_text(
        config.get("resume_instruction"),
        "If there was a previous session, continue naturally. Otherwise start simply.",
    )

    return f"""
{static_agent_role(agent_type)}

Teach naturally in real-time spoken conversation.

Learner profile:
- Native language: {native_language}
- Target language: {target_language}
- Level: {learner_level}
- Interests: {learner_interests_text}
- Strengths: {learner_strengths_text}
- Weaknesses: {learner_weaknesses_text}

Lesson context:
- Topic: {lesson_topic}
- Goal: {lesson_goal}

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


def count_remote_agents(room: rtc.Room) -> int:
    count = 0
    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            count += 1
    return count


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
    runtime = ReplyOrchestrator()
    room_name = "unknown"
    room_lock: Optional[PgRoomLock] = None

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

        system_prompt = build_system_prompt(agent_type, config)
        opening_line = clean_text(config.get("opening_line"), "Hello. Let's continue from where we left off.")
        voice = random.choice(VOICE_MAP.get(agent_type, VOICE_MAP[DEFAULT_AGENT_TYPE]))

        await ctx.connect()
        room_name = getattr(ctx.room, "name", "unknown")
        logger.info("agent_entry room=%s agent_type=%s voice=%s", room_name, agent_type, voice)

        room_lock = PgRoomLock(room_name)
        locked = await room_lock.acquire()
        if not locked:
            logger.warning("duplicate_agent_pg_lock room=%s", room_name)
            ctx.shutdown("duplicate_agent_pg_lock")
            return

        stale_count = await wait_for_stale_agents_to_leave(ctx.room)
        if stale_count > 0:
            logger.warning("room=%s still has %s active agent(s); shutting down duplicate", room_name, stale_count)
            ctx.shutdown("room_already_has_agent")
            return

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

        await runtime.start(session)
        await runtime.send_opening(opening_line)

        shutdown_event = asyncio.Event()

        async def _on_shutdown():
            try:
                logger.info("ctx shutdown callback triggered room=%s", room_name)
                shutdown_event.set()
            except Exception:
                logger.exception("shutdown callback failed room=%s", room_name)

        add_shutdown_callback = getattr(ctx, "add_shutdown_callback", None)
        if callable(add_shutdown_callback):
            add_shutdown_callback(_on_shutdown)
            await shutdown_event.wait()
        else:
            while True:
                await asyncio.sleep(2.0)

    except asyncio.CancelledError:
        logger.warning("agent_cancelled room=%s", room_name)
        raise
    except Exception as e:
        logger.exception("agent_error room=%s error=%s", room_name, e)
        raise
    finally:
        logger.info("agent_cleanup_start room=%s", room_name)

        with contextlib.suppress(Exception):
            await runtime.stop()

        if session is not None:
            close_method = getattr(session, "aclose", None)
            if close_method is not None:
                with contextlib.suppress(Exception):
                    await close_method()

        if room_lock is not None:
            with contextlib.suppress(Exception):
                await room_lock.release()

        logger.info("agent_cleanup_done room=%s", room_name)


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)