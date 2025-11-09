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
        You are a friendly onboarding guide who helps new users understand how to use the Zabano platform.
        Speak in Persian and use simple English examples when relevant.
        Keep responses short, warm, and motivating.
        """,
        "voice_choices": ["nova", "shimmer"],
    },
    "assessment": {
        "instructions": """
        You are an English proficiency assessor.
        Conduct a short conversation to evaluate user's English speaking and comprehension.
        Ask open questions, rate them privately (don’t show scores to user).
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
    """Entry point for the agent."""

    # Get the agent type from job metadata (set by Django backend)
    agent_type = ctx.room.name.split("-")[-1]

    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])

    # Pick a random voice
    voice = random.choice(config["voice_choices"])

    # Custom STT to force transcription (not translation)
    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *args, **kwargs):
            kwargs["task"] = "transcribe"
            kwargs.pop("translate", False)
            return await super().transcribe(*args, **kwargs)

        # Setup session with components

    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini")),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    # avatar = simli.AvatarSession(
    #     simli_config=simli.SimliConfig(
    #         api_key=os.getenv("SIMLI_API_KEY"),
    #         face_id=os.getenv("SIMLI_FACE_ID", "cace3ef7-a4c4-425d-a8cf-a5358eb0c427"),
    #     ),
    # )
    #
    # await avatar.start(session, room=ctx.room)

    # Start the session
    await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))

    # Optional initial message
    await session.generate_reply(instructions=f"شروع گفتگو: نوع عامل شما: {agent_type}")


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
