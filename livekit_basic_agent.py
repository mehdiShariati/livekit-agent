import json
import random
import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
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
    },
    "assessment": {
        "instructions": """
        You are an English proficiency assessor.
        Conduct a short conversation to evaluate user's English speaking and comprehension.
        Ask open questions, rate them privately (don't show scores to user).
        Speak partly in English, partly in Persian.
        """,
        "voice_choices": ["coral", "verse"],
    },
    "tutor": {
        "instructions": """
        You are an expert English tutor for Persian speakers.
        Always explain grammar in Persian and show clear English examples.
        Be kind, interactive, and patient.
        """,
        "voice_choices": ["nova", "coral"],
    },
}


# ---------------------------------------------
# 👩‍🏫 Dynamic Assistant class
# ---------------------------------------------
class DynamicAssistant(Agent):
    def __init__(self, agent_type="tutor"):
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        super().__init__(instructions=config["instructions"])
        self.agent_type = agent_type


# ---------------------------------------------
# 🚀 Entrypoint
# ---------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    metadata = {}
    raw_meta = None

    # Try multiple ways to access metadata
    print(f"🔍 JobContext type: {type(ctx.job)}")
    print(f"🔍 JobContext dir: {dir(ctx.job)}")

    # Method 1: Check agent_dispatch
    if hasattr(ctx.job, 'agent_dispatch') and ctx.job.agent_dispatch:
        raw_meta = ctx.job.agent_dispatch.metadata
        print(f"📦 Found metadata via agent_dispatch: {raw_meta}")

    # Method 2: Check direct metadata attribute
    elif hasattr(ctx.job, 'metadata'):
        raw_meta = ctx.job.metadata
        print(f"📦 Found metadata via job.metadata: {raw_meta}")

    # Method 3: Check job info
    elif hasattr(ctx.job, 'job'):
        if hasattr(ctx.job.job, 'metadata'):
            raw_meta = ctx.job.job.metadata
            print(f"📦 Found metadata via job.job.metadata: {raw_meta}")

    print(f"🔍 Raw metadata type: {type(raw_meta)}")
    print(f"🔍 Raw metadata value: {raw_meta}")

    # Parse metadata
    if raw_meta:
        try:
            if isinstance(raw_meta, str):
                metadata = json.loads(raw_meta)
            elif isinstance(raw_meta, dict):
                metadata = raw_meta
            else:
                print(f"⚠️ Unexpected metadata type: {type(raw_meta)}")
        except Exception as e:
            print(f"❌ Metadata parse error: {e}")
            print(f"   Raw value: {raw_meta}")

    print(f"🧩 Parsed metadata: {metadata}")

    # Check if this is a zabano job
    if metadata.get("source") != "zabano":
        print(f"⚠️ Ignoring non-zabano job. Metadata: {metadata}")
        # If metadata is completely empty, still proceed (for testing)
        if not metadata:
            print("⚠️ Empty metadata - proceeding with default agent_type")
            agent_type = "tutor"
        else:
            return
    else:
        # Get agent type from metadata
        agent_type = metadata.get("agent_type", "tutor")

    print(f"✅ Starting agent with type: {agent_type}")

    # Get configuration
    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])

    # Pick a random voice
    voice = random.choice(config["voice_choices"])
    print(f"🎤 Selected voice: {voice}")

    # Custom STT to force transcription (not translation)
    class CustomWhisperSTT(openai.STT):
        async def _recognize_impl(self, buffer, *, language=None):
            """Override to force transcription mode"""
            return await super()._recognize_impl(
                buffer,
                language=language,
                # Force transcribe, not translate
                prompt="transcribe"
            )

    # Setup session with components
    try:
        session = AgentSession(
            stt=CustomWhisperSTT(model="whisper-1"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        print("🔧 Session components initialized")

        # Uncomment if using Simli avatar
        # avatar = simli.AvatarSession(
        #     simli_config=simli.SimliConfig(
        #         api_key=os.getenv("SIMLI_API_KEY"),
        #         face_id=os.getenv("SIMLI_FACE_ID", "cace3ef7-a4c4-425d-a8cf-a5358eb0c427"),
        #     ),
        # )
        # await avatar.start(session, room=ctx.room)

        # Start the session
        await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))
        print(f"✅ Session started in room: {ctx.room.name}")

        # Optional initial message
        greeting_map = {
            "onboarding": "سلام! به زبانو خوش آمدید. چطور می‌تونم کمکتون کنم؟",
            "assessment": "Hello! سلام! Ready to test your English? آماده‌اید؟",
            "tutor": "سلام! من معلم انگلیسی شما هستم. بیایید شروع کنیم!",
        }

        greeting = greeting_map.get(agent_type, greeting_map["tutor"])
        await session.generate_reply(instructions=greeting)
        print(f"👋 Sent greeting for {agent_type}")

    except Exception as e:
        print(f"❌ Session initialization error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)