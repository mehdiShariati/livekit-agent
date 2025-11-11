import json
import random
import os
import asyncio
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv(".env")

# -----------------------------
# Agent Templates
# -----------------------------
AGENT_TYPES = {
    "onboarding": {
        "instructions": "You are a friendly onboarding guide who helps new users understand zabano.com. Speak in Persian. Keep responses short and warm.",
        "voice_choices": ["nova"],
        "greeting": "سلام! به زبانو خوش آمدید. چطور می‌تونم کمکتون کنم؟"
    },
    "assessment": {
        "instructions": "You are an English proficiency assessor. Speak partly in English, partly in Persian. Ask open questions, rate privately.",
        "voice_choices": ["coral", "verse"],
        "greeting": "Hello! سلام! Ready to test your English? آماده‌اید؟"
    },
    "tutor": {
        "instructions": "You are an expert English tutor for Persian speakers. Explain grammar in Persian with English examples. Be kind and patient.",
        "voice_choices": ["nova", "coral"],
        "greeting": "سلام! من معلم انگلیسی شما هستم. بیایید شروع کنیم!"
    },
}


# -----------------------------
# Dynamic Assistant
# -----------------------------
class DynamicAssistant(Agent):
    def __init__(self, agent_type="tutor"):
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        super().__init__(instructions=config["instructions"])
        self.agent_type = agent_type


# -----------------------------
# Entrypoint
# -----------------------------
async def entrypoint(ctx: agents.JobContext):
    print(f"🚀 Agent starting in room: {ctx.room.name}")

    # Parse metadata
    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(f"📦 Job metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    agent_type = metadata.get("agent_type", "tutor")
    behavior = metadata.get('config', {}).get('behavior')

    # Connect to room
    await ctx.connect()
    await asyncio.sleep(0.5)

    # Avoid duplicate agents
    agent_count = sum(
        1 for p in ctx.room.remote_participants.values()
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )
    if agent_count > 0:
        print(f"⚠️ {agent_count} agent(s) already present, skipping start")
        return

    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
    voice = random.choice(config["voice_choices"])

    # Force whisper to transcribe
    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *args, **kwargs):
            kwargs["task"] = "transcribe"
            kwargs.pop("translate", None)
            return await super().transcribe(*args, **kwargs)

    # Transcription + LLM hooks
    async def on_transcription(text: str):
        print("🎙️ STT:", text)

    async def on_llm_output(text: str):
        print("🤖 LLM:", text)

    # Session
    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    # Start
    await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))

    # ✅ Hook STT / LLM events (old API)
    @session.on("user_input_transcribed")
    async def _on_user_input(ev):
        text = ev.text
        print("🎙️ STT:", text)
        await on_transcription(text)

    @session.on("conversation_item_added")
    async def _on_llm_output(ev):
        if ev.role == "assistant" and ev.type == "output_text":
            text = ev.text
            print("🤖 LLM:", text)
            await on_llm_output(text)

    # Greeting
    greeting_text = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")
    if behavior:
        greeting_text = behavior.get("text", json.dumps(behavior)) if isinstance(behavior, dict) else str(behavior)

    await session.generate_reply(instructions=greeting_text)
    print(f"✅ {agent_type} agent started successfully")

    await ctx.wait_for_participant()

