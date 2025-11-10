import json
import random
import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

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

# Track active sessions to prevent duplicates
active_sessions = set()


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
    """Main entrypoint for the LiveKit agent."""

    # Parse metadata
    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(metadata)
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
    instruction = metadata.get('livekit')
    behavior = ""
    if instruction:
        behavior = instruction.get('behavior')
    print(f"behavior is {behavior}")
    # Prevent duplicate sessions
    session_key = f"{ctx.room.name}_{agent_type}"
    if session_key in active_sessions:
        print(f"⚠️ Session already active: {session_key}")
        return

    active_sessions.add(session_key)

    try:
        # Get configuration
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        voice = random.choice(config["voice_choices"])

        print(f"✅ Starting {agent_type} agent in room {ctx.room.name} with voice {voice}")

        # Custom STT to force transcription (not translation)
        class CustomWhisperSTT(openai.STT):
            async def _recognize_impl(self, buffer, *, language=None):
                """Override to force transcription mode."""
                return await super()._recognize_impl(
                    buffer,
                    language=language,
                    prompt="transcribe"
                )

        # Setup session components
        session = AgentSession(
            stt=CustomWhisperSTT(model="whisper-1"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # Start the session
        await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))
        greeting = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")

        # Send greeting
        if behavior:
            greeting = json.dumps(behavior)
        await session.generate_reply(instructions=greeting)

        print(f"✅ {agent_type} agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up session tracking
        active_sessions.discard(session_key)


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)