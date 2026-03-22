import json
import random
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero, simli

# Load environment variables
load_dotenv(".env")

# ---------------------------------------------
# 🧱 Agent Template Configuration
# ---------------------------------------------
AGENT_TYPES = {
    "onboarding": {
        "instructions": """
        You are a friendly onboarding guide who helps new users understand how to use the zabano.com platform.
        Speak in Persian.
        Keep responses short, warm, and motivating.
        """,
        "voice_choices": ["nova"],
        "greeting": "سلام! به زبانو خوش آمدید. چطور می‌تونم کمکتون کنم؟"
    },
    "assessment": {
        "instructions": """
        You are an English proficiency assessor.
        Conduct a short conversation to evaluate user's English speaking and comprehension.
        Ask open questions, rate them privately (don't show scores to user).
        Speak partly in English, partly in Persian.
        """,
        "voice_choices": ["coral", "verse"],
        "greeting": "Hello! سلام! Ready to test your English? آماده‌اید؟"
    },
    "tutor": {
        "instructions": """
        You are an expert English tutor for Persian speakers.
        Always explain grammar in Persian and show clear English examples.
        Be kind, interactive, and patient.
        """,
        "voice_choices": ["nova", "coral"],
        "greeting": "سلام! من معلم انگلیسی شما هستم. بیایید شروع کنیم!"
    },
}


# ---------------------------------------------
# 👩‍🏫 Dynamic Assistant class
# ---------------------------------------------
class DynamicAssistant(Agent):
    def __init__(self, agent_type="tutor", transcript_context: Optional[str] = None):
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        instructions = config["instructions"]
        extra = (transcript_context or "").strip()
        if extra:
            instructions = (
                f"{instructions}\n\n"
                "Prior conversation (same room / resumed session). Continue naturally; "
                "do not restart the lesson from scratch unless the learner asks.\n"
                f"{extra[-8000:]}"
            )
        super().__init__(instructions=instructions)
        self.agent_type = agent_type


def _count_remote_agents(room: rtc.Room) -> int:
    n = 0
    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            n += 1
    return n


async def _wait_for_stale_agents_to_leave(
    room: rtc.Room, *, max_wait_s: float = 3.5, poll_s: float = 0.35
) -> int:
    """
    After dispatch, LiveKit may still list a disconnecting agent briefly, or a resume may race
    with the previous session. If count is 0 immediately, proceed. If agents are present, poll
    until they clear or timeout; only skip starting when agents remain after the wait.
    """
    count = _count_remote_agents(room)
    if count == 0:
        return 0
    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_wait_s
    while count > 0 and loop.time() < deadline:
        await asyncio.sleep(poll_s)
        count = _count_remote_agents(room)
    return count


# ---------------------------------------------
# 🚀 Entrypoint
# ---------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the LiveKit agent."""

    # Parse metadata
    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(f"📦 Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    # Validate this is a zabano job
    if metadata.get("source") != "zabano":
        if not metadata:
            # Empty metadata - use default for testing
            print("⚠️ No metadata provided, using default tutor agent")
            agent_type = "tutor"
        else:
            print(f"⚠️ Ignoring non-zabano job: {metadata}")
            return
    else:
        agent_type = metadata.get("agent_type", "tutor")

    instruction = metadata.get('config')
    behavior = ""
    if instruction:
        behavior = instruction.get('behavior')

    transcript_context = ""
    if isinstance(instruction, dict):
        for key in ("chat_history", "conversation_history", "conversation_text"):
            val = instruction.get(key)
            if isinstance(val, str) and val.strip():
                transcript_context = val.strip()
                break

    # Connect to room
    await ctx.connect()

    agent_count = await _wait_for_stale_agents_to_leave(ctx.room)
    if agent_count > 0:
        print(
            f"⚠️ {agent_count} agent(s) still in room {ctx.room.name} after stale wait — "
            "skipping duplicate worker"
        )
        ctx.shutdown("room_already_has_agent")
        return

    print(f"✅ No active agent in room {ctx.room.name}, proceeding to start {agent_type} agent")

    try:
        # Get configuration
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        voice = random.choice(config["voice_choices"])

        print(f"✅ Starting {agent_type} agent in room {ctx.room.name} with voice {voice}")

        # Custom STT to force transcription (not translation)
        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                # Force Whisper to transcribe (not translate)
                kwargs["task"] = "transcribe"  # 👈 critical flag
                kwargs.pop("translate", False)  # remove translation if passed accidentally
                return await super().transcribe(*args, **kwargs)

        # Setup session components
        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # avatar = simli.AvatarSession(
        #     simli_config=simli.SimliConfig(
        #         api_key=os.getenv("SIMLI_API_KEY"),
        #         face_id="14de6eb1-0ea6-4fde-9522-8552ce691cb6",
        #         # ID of the Simli face to use for your avatar. See "Face setup" for details.
        #     ),
        # )

        # Start the avatar and wait for it to join
        # await avatar.start(session, room=ctx.room)

        # Start the session
        await session.start(
            room=ctx.room,
            agent=DynamicAssistant(agent_type, transcript_context=transcript_context or None),
        )
        greeting = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")

        # Send greeting
        if behavior:
            greeting = json.dumps(behavior)  # Don't stringify it, use it directly

        await session.generate_reply(instructions=greeting)

        print(f"✅ {agent_type} agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)