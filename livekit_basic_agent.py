import json
import random
import os
import requests
import asyncio
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
    def __init__(self, agent_type="tutor"):
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        super().__init__(instructions=config["instructions"])
        self.agent_type = agent_type


def send_to_backend(payload):
    url = "https://api.zabano.com/api/livekit/webhook/"
    headers = {
        'sec-ch-ua-platform': '"Linux"',
        'Referer': 'https://zabano.com/',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?0',
        'Content-Type': 'application/json'
    }

    requests.request("POST", url, headers=headers, data=payload)


def log_to_file(room_name, role, message):
    """Append chat messages to a text file per room."""
    os.makedirs("chat_logs", exist_ok=True)
    file_path = os.path.join("chat_logs", f"{room_name}.txt")

    formatted_message = f"{role}: {message}\n"

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(formatted_message)


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

    # Connect to room
    await ctx.connect()

    # Wait a bit for other agents to appear (handle race condition)
    await asyncio.sleep(0.5)

    # Check if there are already agents in the room
    participants = ctx.room.remote_participants
    agent_count = 0
    print(participants)
    for participant in participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            agent_count += 1
            print(f"⚠️ Found existing agent in room: {participant.identity}")

    if agent_count > 0:
        print(f"⚠️ {agent_count} agent(s) already in room {ctx.room.name}, skipping")
        return

    print(f"✅ No existing agent found, proceeding to start {agent_type} agent")

    try:
        # Get configuration
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        voice = random.choice(config["voice_choices"])

        print(f"✅ Starting {agent_type} agent in room {ctx.room.name} with voice {voice}")

        # Custom STT to force transcription (not translation)
        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                # Force Whisper to transcribe (not translate)
                kwargs["task"] = "transcribe"
                kwargs.pop("translate", False)
                return await super().transcribe(*args, **kwargs)

        # Setup session components
        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # ---------------------------------------------
        # 🧹 Auto-cleanup when user leaves the room
        # ---------------------------------------------
        async def on_participant_disconnected(ev):
            participant = ev.participant
            print(f"👋 Participant left: {participant.identity}")

            # Ignore if another agent disconnects
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return

            print("🛑 User left — stopping session and disconnecting agent...")

            try:
                await session.stop()
            except Exception as e:
                print("Error stopping session:", e)

            try:
                await ctx.room.disconnect()
            except Exception as e:
                print("Error disconnecting room:", e)

        ctx.room.on("participant_disconnected", on_participant_disconnected)

        # Async handlers
        async def on_transcription(text: str):
            print("🎙️ STT:", text)

        async def on_llm_output(text: str):
            print("🤖 LLM:", text)

        def _wrap_on_transcription(ev):
            asyncio.create_task(on_transcription(ev.transcript))

        def _wrap_on_llm_output(ev):
            if hasattr(ev.item, "role"):
                try:
                    if ev.item.role == "assistant":
                        role = "agent"
                    elif ev.item.role == "user":
                        role = "user"
                    else:
                        role = "system"

                    room_name = getattr(ctx.room, "name", "default_room")
                    message = ev.item.content

                    if isinstance(message, list):
                        message = " ".join(str(m) for m in message)
                    elif not isinstance(message, str):
                        message = str(message)

                    message = message.strip()
                    log_to_file(room_name, role, message)

                except Exception as e:
                    print("Error logging message:", e)

        session.on("user_input_transcribed", _wrap_on_transcription)
        session.on("conversation_item_added", _wrap_on_llm_output)

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
        await session.start(room=ctx.room, agent=DynamicAssistant(agent_type))
        greeting = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")

        if behavior:
            greeting = json.dumps(behavior)

        await session.generate_reply(instructions=greeting)

        print(f"✅ {agent_type} agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
