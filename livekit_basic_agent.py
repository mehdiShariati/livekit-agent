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

    # Connect to the room
    await ctx.connect()
    await asyncio.sleep(0.5)  # wait for other agents

    # Check for existing agents
    agent_count = sum(1 for p in ctx.room.remote_participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    if agent_count > 0:
        print(f"⚠️ {agent_count} agent(s) already present, skipping start")
        return

    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
    voice = random.choice(config["voice_choices"])

    # Custom STT forcing transcription
    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *args, **kwargs):
            kwargs["task"] = "transcribe"
            kwargs.pop("translate", False)
            return await super().transcribe(*args, **kwargs)

    # Setup session
    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    # Start agent session
    await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))

    # Prepare greeting
    greeting_text = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")
    if behavior:
        if isinstance(behavior, dict):
            greeting_text = behavior.get("text", json.dumps(behavior))
        else:
            greeting_text = str(behavior)

    # Send greeting
    await session.generate_reply(instructions=greeting_text)
    print(f"✅ {agent_type} agent started successfully")

    # -----------------------------
    # Register all major events
    # -----------------------------
    def sync_task(fn, *args, **kwargs):
        """Helper to run async code safely in sync callbacks"""
        asyncio.create_task(fn(*args, **kwargs))

    @ctx.room.on("participant_connected")
    def participant_connected(participant: rtc.RemoteParticipant):
        print(f"👤 Participant joined: {participant.identity}")

    @ctx.room.on("participant_disconnected")
    def participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"👤 Participant left: {participant.identity}")

    @ctx.room.on("transcription_received")
    def transcription_received(segments, participant, publication):
        text = " ".join(seg.text for seg in segments)
        print(f"🗣 Transcription from {participant.identity}: {text}")

    @ctx.room.on("data_received")
    def data_received(data_packet):
        print(f"📨 Data received: {data_packet}")

    @ctx.room.on("active_speakers_changed")
    def active_speakers_changed(speakers):
        identities = [s.identity for s in speakers]
        print(f"🎤 Active speakers: {identities}")

    # You can add additional events like track_published, track_unpublished, etc. similarly:
    # @ctx.room.on("track_published")
    # def track_published(publication, participant):
    #     ...

    # Keep the agent alive
    await ctx.wait_for_participant()
