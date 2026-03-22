import asyncio
import json
import logging
import os
import random
from typing import Any

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

logger = logging.getLogger("rockonlearn.agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ============================================================
# Static application policy
# ============================================================

DEFAULT_AGENT_TYPE = "tutor"
DEFAULT_LLM = os.getenv("LLM_CHOICE", "gpt-4o-mini")
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"

VOICE_MAP = {
    "tutor": ["nova", "coral"],
    "assessment": ["coral", "verse"],
    "onboarding": ["nova"],
}

_VAD_INSTANCE = None


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

    role_text = static_agent_role(agent_type)
    teaching_policy_text = static_teaching_policy(agent_type)
    language_policy_text = language_policy(native_language, target_language, learner_level)
    safety_rules = static_safety_rules()

    return f"""
{role_text}

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
{language_policy_text}

Teaching policy:
{teaching_policy_text}

Progress summary:
{progress_summary}

Previous session summary:
{memory_summary}

Resume behavior:
{resume_instruction}

Safety and operating rules:
{safety_rules}
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
    max_wait_s: float = 3.5,
    poll_s: float = 0.35,
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


class AgentRuntimeState:
    """
    Prevent overlapping generate_reply() calls.
    This is the main app-side mitigation for TTS stream races.
    """

    def __init__(self) -> None:
        self.reply_lock = asyncio.Lock()
        self.opening_sent = False
        self.shutting_down = False

    async def generate_reply_once(
        self,
        session: AgentSession,
        instructions: str,
        *,
        timeout_s: float = 20.0,
        tag: str = "reply",
    ) -> bool:
        if self.shutting_down:
            logger.info("skip generate_reply because runtime is shutting down tag=%s", tag)
            return False

        text = clean_text(instructions)
        if not text:
            logger.info("skip empty generate_reply tag=%s", tag)
            return False

        async with self.reply_lock:
            if self.shutting_down:
                logger.info("skip locked generate_reply because runtime is shutting down tag=%s", tag)
                return False

            logger.info("reply_start tag=%s text=%r", tag, text[:160])

            try:
                await asyncio.wait_for(
                    session.generate_reply(instructions=text),
                    timeout=timeout_s,
                )
                logger.info("reply_done tag=%s", tag)
                return True
            except asyncio.TimeoutError:
                logger.warning("reply_timeout tag=%s", tag)
                return False
            except asyncio.CancelledError:
                logger.warning("reply_cancelled tag=%s", tag)
                raise
            except Exception as e:
                logger.exception("reply_failed tag=%s error=%s", tag, e)
                return False

    async def send_opening(
        self,
        session: AgentSession,
        opening_line: str,
    ) -> bool:
        if self.opening_sent or self.shutting_down:
            logger.info("opening skipped opening_sent=%s shutting_down=%s", self.opening_sent, self.shutting_down)
            return False

        self.opening_sent = True
        return await self.generate_reply_once(
            session,
            opening_line,
            timeout_s=20.0,
            tag="opening",
        )

    async def shutdown(self) -> None:
        self.shutting_down = True


async def entrypoint(ctx: agents.JobContext):
    metadata: dict[str, Any] = {}

    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
        except Exception as e:
            logger.exception("failed to parse job metadata: %s", e)
            metadata = {}

    if metadata and metadata.get("source") != "rockonlearn":
        logger.info("ignoring non-rockonlearn job metadata=%s", metadata)
        return

    agent_type = clean_text(metadata.get("agent_type"), DEFAULT_AGENT_TYPE)
    config = metadata.get("config") or {}
    if not isinstance(config, dict):
        config = {}

    system_prompt = build_system_prompt(agent_type, config)
    opening_line = clean_text(config.get("opening_line"), "Hello. Let's continue from where we left off.")
    voice = random.choice(VOICE_MAP.get(agent_type, VOICE_MAP[DEFAULT_AGENT_TYPE]))

    logger.info("agent_entry room=%s agent_type=%s voice=%s", getattr(ctx.room, "name", "unknown"), agent_type, voice)

    await ctx.connect()

    agent_count = await wait_for_stale_agents_to_leave(ctx.room)
    if agent_count > 0:
        logger.warning(
            "room=%s still has %s active agent(s); skipping duplicate worker",
            ctx.room.name,
            agent_count,
        )
        ctx.shutdown("room_already_has_agent")
        return

    runtime = AgentRuntimeState()

    try:
        session = AgentSession(
            stt=CustomWhisperSTT(model=DEFAULT_STT_MODEL),
            llm=openai.LLM(model=DEFAULT_LLM),
            tts=openai.TTS(voice=voice),
            vad=get_vad(),
        )

        assistant = DynamicAssistant(instructions=system_prompt)

        await session.start(
            room=ctx.room,
            agent=assistant,
        )

        logger.info("agent_started room=%s", ctx.room.name)

        # Important: only one short opening line.
        # Do not pass serialized config, behavior JSON, or transcript here.
        await runtime.send_opening(session, opening_line)

        # Keep the worker alive while the room/session is active.
        # Depending on your deployment, LiveKit may manage lifecycle already.
        # This loop keeps the coroutine alive without generating extra speech.
        while True:
            await asyncio.sleep(2.0)

    except asyncio.CancelledError:
        logger.warning("agent_entry cancelled room=%s", getattr(ctx.room, "name", "unknown"))
        raise
    except Exception as e:
        logger.exception("agent error room=%s error=%s", getattr(ctx.room, "name", "unknown"), e)
        raise
    finally:
        await runtime.shutdown()
        logger.info("agent_cleanup room=%s", getattr(ctx.room, "name", "unknown"))


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)