import json
import random
import os
import asyncio
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero
from livekit.plugins.openai import realtime
from livekit.plugins.openai.realtime import TurnDetection

load_dotenv(".env")


def _cfg_str(cfg: dict, key: str, default: str = "") -> str:
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return default
    s = str(v).strip()
    return s if s else default


def _is_onboarding_speaking(metadata: dict) -> bool:
    """First speaking check in app onboarding / placement (pre-login onb-* flow)."""
    agent_type = str(metadata.get("agent_type") or "").lower()
    cfg = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    assess = str(cfg.get("assessment_type") or "").lower()
    if agent_type not in ("onboarding", "assessment"):
        return False
    return assess in ("onboarding", "placement", "")


def _build_onboarding_dynamic_instructions(cfg: dict) -> str:
    """Structured context for system prompt — never JSON for the user."""
    native = _cfg_str(cfg, "native_language", "unknown")
    target = _cfg_str(cfg, "speaking_language") or _cfg_str(cfg, "target_language", "English")
    learning = _cfg_str(cfg, "learning_language", target)
    level = _cfg_str(cfg, "learner_level") or _cfg_str(cfg, "user_level", "A1")
    goal = _cfg_str(cfg, "selected_goal", "")
    goal_focus = _cfg_str(cfg, "goal_focus", "")
    profile = _cfg_str(cfg, "progress_summary", "")[:2500]

    lines = [
        "## Learner context (internal — do not read aloud as a list)",
        f"- Native language (for occasional support): {native}",
        f"- Language they are learning / should practice speaking: {learning}",
        f"- Rough self-reported level hint: {level} (adapt complexity; do not quiz them on labels)",
    ]
    if goal:
        lines.append(f"- Stated motivation / goal: {goal}")
    if goal_focus:
        lines.append(f"- Goal focus: {goal_focus}")
    if profile:
        lines.append(f"- Profile notes: {profile}")
    return "\n".join(lines)


ONBOARDING_SPEAKING_SYSTEM = """
You are RockOn's voice tutor for a **very short first speaking check** (about one minute total).

Personality: warm, sharp, human — like a great language coach, not customer support and not an exam proctor.

Hard rules:
- Total session target: **60–90 seconds** of back-and-forth. Be concise.
- Ask **at most 2 questions** in the whole session. Question 1 (required): ask why they are learning {target_language} — natural wording, in {target_language} when possible.
- Optional question 2: only if their answer was too short or unclear — one short follow-up (e.g. "What do you want to use it for?").
- After they have answered in a meaningful way, **stop asking questions**.
- Then give **brief closing feedback** in this order:
  1) One specific positive (e.g. clarity, confidence, vocabulary).
  2) **Exactly one** small correction (grammar or phrasing) — say what to say instead, briefly.
  3) A rough level estimate in plain words (e.g. "around A2" or "between A2 and B1") — not a lecture.
  4) One motivating line about improving with daily practice on RockOn.
  5) Close clearly so they know the check is done (e.g. "That's your quick check — nice work.").
- Do **not** say: "How can I help you?", "Welcome to the platform", "Ready to test your English?", or similar generic support lines.
- Do **not** list many corrections, long paragraphs, or internal scores.
- Encourage them to speak in {target_language}; use {native_language} only briefly for comfort if needed.
- Multilingual input: if they mix languages, understand and gently steer back to {target_language}.

Opening: follow the separate first-turn instruction you receive after joining — it tells you exactly how to start.
""".strip()


GENERIC_AGENT_TYPES = {
    "onboarding": {
        "instructions": """
        You are a friendly onboarding guide for RockOn (language learning).
        Keep responses short, warm, and motivating. Do not sound like customer support.
        """,
        "voice_choices": ["nova", "coral"],
        "greeting_instruction": "Greet briefly and invite them to start their quick speaking check in a friendly way.",
    },
    "onboarding_assessment": {
        "instructions": ONBOARDING_SPEAKING_SYSTEM,
        "voice_choices": ["coral", "verse", "nova"],
        "greeting_instruction": None,
    },
    "assessment": {
        "instructions": """
        You are a speaking coach for RockOn. Run a short, natural conversation to sample the learner's level.
        Be warm and efficient; avoid exam-like tone.
        """,
        "voice_choices": ["coral", "verse"],
        "greeting_instruction": "Open with a short warm line, then ask one open question about their learning goals.",
    },
    "tutor": {
        "instructions": """
        You are an expert language tutor on RockOn. Be kind, interactive, and concise.
        Match the learner's level and encourage speaking in the target language.
        """,
        "voice_choices": ["nova", "coral"],
        "greeting_instruction": "Greet warmly and start the lesson topic naturally.",
    },
}


class DynamicAssistant(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)
        self.instructions_text = instructions


def _resolve_agent_config(metadata: dict) -> tuple[str, dict]:
    """Returns (voice_key_for_lookup, config dict with instructions + greeting_instruction)."""
    raw_type = str(metadata.get("agent_type") or "tutor").lower()
    cfg = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}

    if _is_onboarding_speaking(metadata):
        target = _cfg_str(cfg, "speaking_language") or _cfg_str(cfg, "target_language", "English")
        native = _cfg_str(cfg, "native_language", "their native language")
        dynamic = _build_onboarding_dynamic_instructions(cfg)
        base = ONBOARDING_SPEAKING_SYSTEM.format(target_language=target, native_language=native)
        full_instructions = f"{base}\n\n{dynamic}"
        return "onboarding_assessment", {
            "instructions": full_instructions,
            "voice_choices": GENERIC_AGENT_TYPES["onboarding_assessment"]["voice_choices"],
            "greeting_instruction": None,
        }

    conf = GENERIC_AGENT_TYPES.get(raw_type, GENERIC_AGENT_TYPES["tutor"])
    return raw_type, conf


async def entrypoint(ctx: agents.JobContext):
    metadata = {}
    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(f"Metadata keys: {list(metadata.keys())}")
        except Exception as e:
            print(f"Failed to parse metadata: {e}")

    if metadata.get("source") != "zabano":
        if not metadata:
            print("No metadata — default tutor")
            metadata = {"source": "zabano", "agent_type": "tutor", "config": {}}
        else:
            print(f"Ignoring non-zabano job: {metadata.get('source')}")
            return

    await ctx.connect()
    await asyncio.sleep(0.5)

    participants = ctx.room.remote_participants
    agent_count = sum(
        1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )
    if agent_count > 0:
        print(f"{agent_count} agent(s) already in room, skipping")
        return

    voice_key, type_conf = _resolve_agent_config(metadata)
    voice = random.choice(type_conf["voice_choices"])
    print(f"Starting agent mode={voice_key!r} voice={voice} room={ctx.room.name}")

    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *args, **kwargs):
            kwargs["task"] = "transcribe"
            kwargs.pop("translate", None)
            return await super().transcribe(*args, **kwargs)

    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=realtime.RealtimeModel(
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="medium",
                create_response=True,
                interrupt_response=True,
            ),
        ),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    agent = DynamicAssistant(type_conf["instructions"])
    await session.start(room=ctx.room, agent=agent)

    cfg = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    opening_line = _cfg_str(cfg, "opening_line", "")

    if voice_key == "onboarding_assessment" or _is_onboarding_speaking(metadata):
        target = _cfg_str(cfg, "speaking_language") or _cfg_str(cfg, "target_language", "English")
        native = _cfg_str(cfg, "native_language", "")
        first_turn = opening_line or (
            f"Begin the session now. First spoken turn only: one short warm line"
            + (f" (you may use {native} for that single line if natural)" if native else "")
            + f", then switch to {target}. Say you're doing a quick speaking check (~1 minute), not a test. "
            f"Then ask why they are learning {target} — one clear question. Keep under ~25 seconds of speech, then stop and listen."
        )
        await session.generate_reply(instructions=first_turn)
    else:
        legacy_behavior = cfg.get("behavior")
        if isinstance(legacy_behavior, str) and legacy_behavior.strip():
            await session.generate_reply(instructions=legacy_behavior.strip())
        elif isinstance(legacy_behavior, dict):
            print("Ignoring config.behavior dict — use string instructions or opening_line only")
            gi = type_conf.get("greeting_instruction") or "Greet the user briefly and naturally."
            await session.generate_reply(instructions=gi)
        elif opening_line.strip():
            await session.generate_reply(instructions=opening_line)
        else:
            gi = type_conf.get("greeting_instruction") or "Greet the user briefly and naturally."
            await session.generate_reply(instructions=gi)

    print(f"Agent started successfully ({voice_key})")


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
