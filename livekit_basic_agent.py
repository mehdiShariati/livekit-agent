import json
import random
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero, simli

# Load environment variables
load_dotenv(".env")

# ----------------------------
# Configuration
# ----------------------------
DJANGO_URL = os.getenv("DJANGO_API_BASE", "https://your-django.com")  # Your Django API base

AGENT_TYPES = {
    "onboarding": {
        "instructions": """
        You are a friendly onboarding guide who helps new users understand how to use the zabano.com platform.
        Speak in Persian. Keep responses short, warm, and motivating.
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


# ----------------------------
# Dynamic Assistant
# ----------------------------
class DynamicAssistant(Agent):
    def __init__(self, agent_type="tutor"):
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        super().__init__(instructions=config["instructions"])
        self.agent_type = agent_type


# ----------------------------
# Helper: Send events to Django
# ----------------------------
async def send_to_django(path, payload):
    print(payload)
    print("-----------------------------")
    # url = f"{DJANGO_URL}{path}"
    # async with aiohttp.ClientSession() as session:
    #     try:
    #         await session.post(url, json=payload)
    #     except Exception as e:
    #         print(f"❌ Failed sending to Django: {e}")


# ----------------------------
# Entrypoint
# ----------------------------
async def entrypoint(ctx: agents.JobContext):
    # --- Parse metadata ---
    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    if metadata.get("source") != "zabano":
        print("⚠️ Not a zabano job, skipping")
        return

    agent_type = metadata.get("agent_type", "tutor")
    behavior = metadata.get("config", {}).get("behavior", "")

    # --- Connect to room ---
    await ctx.connect()
    await asyncio.sleep(0.5)

    # Prevent multiple agents
    for p in ctx.room.remote_participants.values():
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            print(f"⚠️ Agent already exists in room {ctx.room.name}, skipping")
            return

    # --- Session setup ---
    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
    voice = random.choice(config["voice_choices"])

    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *args, **kwargs):
            kwargs["task"] = "transcribe"
            kwargs.pop("translate", False)
            return await super().transcribe(*args, **kwargs)

    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    agent_id = ctx.job.id
    room_name = ctx.room.name

    # --- Register agent instance in Django ---
    send_to_django("/agent-instance/create/", {
        "dispatch_id": agent_id,
        "room_name": room_name,
        "agent_type": agent_type,
    })

    # ----------------------------
    # AgentSession event handlers
    # ----------------------------
    @session.on("assistant_message")
    def on_ai_message(ev):
        print(f"🤖 AI: {ev.text}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "assistant_message",
            "text": ev.text
        })

    @session.on("user_message")
    def on_user_message(ev):
        print(f"🧑 User: {ev.text}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "user_message",
            "text": ev.text
        })

    @session.on("transcription")
    def on_transcription(ev):
        print(f"📝 Transcription chunk: {ev.text}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "transcription",
            "text": ev.text
        })

    @session.on("session_started")
    def on_session_started(ev):
        print("🚀 Session started")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "agent_started"
        })

    @session.on("session_ended")
    def on_session_ended(ev):
        print("🛑 Session ended")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "agent_finished"
        })

    @session.on("participant_joined")
    def on_participant_joined(ev):
        print(f"➕ Participant joined: {ev.participant.identity}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "participant_joined",
            "participant": ev.participant.identity
        })

    @session.on("participant_left")
    def on_participant_left(ev):
        print(f"➖ Participant left: {ev.participant.identity}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "participant_left",
            "participant": ev.participant.identity
        })

    @session.on("error")
    def on_error(ev):
        print(f"❌ Session error: {ev}")
        send_to_django("/livekit/agent-event/", {
            "room": room_name,
            "agent_id": agent_id,
            "type": "error",
            "payload": str(ev)
        })

    # ----------------------------
    # Optional Simli Avatar Events
    # ----------------------------
    # avatar = simli.AvatarSession(
    #     simli_config=simli.SimliConfig(
    #         api_key=os.getenv("SIMLI_API_KEY"),
    #         face_id="14de6eb1-0ea6-4fde-9522-8552ce691cb6"
    #     )
    # )
    # await avatar.start(session, room=ctx.room)
    # @avatar.on("avatar_state_changed")
    # def on_avatar_state(ev):
    #     print(f"🎭 Avatar event: {ev}")
    #     (send_to_django("/livekit/agent-event/", {
    #         "room": room_name,
    #         "agent_id": agent_id,
    #         "type": "avatar_event",
    #         "payload": str(ev)
    #     }))

    # --- Start session and send greeting ---
    await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))
    greeting = behavior or config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")

    # If behavior is a dict, convert to string
    if isinstance(greeting, dict):
        greeting = json.dumps(greeting)

    await session.generate_reply(instructions=greeting)

    print(f"✅ {agent_type} agent started successfully in room {room_name}")


# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
