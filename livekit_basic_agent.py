import json
import random
import os
import re
import asyncio
from datetime import datetime
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
        You are a language proficiency assessor.
        Conduct a short conversation to evaluate user's {{language}} speaking and comprehension.
        Ask open questions, rate them privately (don't show scores to user).
        Speak partly in {{language}} and partly in user's native language.
        """,
        "voice_choices": ["coral", "verse"],
        "greeting": "Hello! سلام! Ready to test your {{language}}? آماده‌اید؟"
    },
    "tutor": {
        "instructions": """
        You are an expert {{language}} tutor for Persian speakers.
        Always explain grammar in Persian and show clear {{language}} examples.
        Be kind, interactive, and patient.
        """,
        "voice_choices": ["nova", "coral"],
        "greeting": "سلام! من معلم {{language}} شما هستم. بیایید شروع کنیم!"
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
# Logging
# ---------------------------------------------
def log_to_file(room_name, role, message):
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

    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(f"📦 Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    if metadata.get("source") != "zabano":
        if not metadata:
            print("⚠️ No metadata provided, using default tutor agent")
            agent_type = "tutor"
        else:
            print(f"⚠️ Ignoring non-zabano job")
            return
    else:
        agent_type = metadata.get("agent_type", "tutor")

    target_language = metadata.get("language", "English")
    instruction = metadata.get("config")
    behavior = None
    if instruction:
        behavior = instruction.get("behavior")

        # Replace all {{language}} placeholders in behavior recursively
        def replace_language(obj):
            if isinstance(obj, dict):
                return {k: replace_language(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_language(v) for v in obj]
            elif isinstance(obj, str):
                return re.sub(r"\{\{\s*language\s*\}\}", target_language, obj)
            else:
                return obj
        behavior = replace_language(behavior)

    # Connect
    await ctx.connect()
    await asyncio.sleep(0.5)

    # Check existing agents
    participants = ctx.room.remote_participants
    agent_count = sum(1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    if agent_count > 0:
        print("⚠️ Existing agent in room — skipping startup")
        return

    print(f"✅ Starting agent type: {agent_type} in language: {target_language}")

    try:
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["tutor"])
        voice = random.choice(config["voice_choices"])

        # Custom Whisper STT to force transcription
        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                kwargs["task"] = "transcribe"
                kwargs.pop("translate", False)
                return await super().transcribe(*args, **kwargs)

        # Create session
        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # ---------------------------------------------
        # Cleanup on user leave
        # ---------------------------------------------
        async def handle_user_left(participant):
            print(f"👋 Participant left: {participant.identity}")
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return
            print("🛑 User left — cleaning up...")
            try: await session.close()
            except Exception as e: print("Error closing session:", e)
            try: await ctx.room.disconnect()
            except Exception as e: print("Error disconnecting room:", e)

        def on_participant_disconnected(participant):
            asyncio.create_task(handle_user_left(participant))

        ctx.room.on("participant_disconnected", on_participant_disconnected)

        # ---------------------------------------------
        # Logging
        # ---------------------------------------------
        async def on_transcription(text: str):
            print("🎙️ STT:", text)

        async def on_llm_output(text: str):
            print("🤖 LLM:", text)

        def _wrap_on_transcription(ev):
            asyncio.create_task(on_transcription(ev.transcript))

        def _wrap_on_llm_output(ev):
            try:
                role = "agent" if ev.item.role == "assistant" else "user"
                content = ev.item.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                log_to_file(ctx.room.name, role, str(content))
            except Exception as e:
                print("Error logging:", e)

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

        # Generate greeting/instructions
        greeting = config.get("greeting", "سلام! چطور می‌تونم کمکتون کنم؟")
        greeting = re.sub(r"\{\{\s*language\s*\}\}", target_language, greeting)

        if behavior:
            instructions_text = f"Target language: {target_language}\n{json.dumps(behavior, ensure_ascii=False)}"
        else:
            instructions_text = greeting

        await session.generate_reply(instructions=instructions_text)
        print("✅ Agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise

# ---------------------------------------------
# Run agent CLI
# ---------------------------------------------
if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
