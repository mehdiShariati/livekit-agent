import json
import os
import re
import random
import asyncio
import time
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

# ----------------------------
# Dynamic Assistant
# ----------------------------
class DynamicAssistant(Agent):
    def __init__(self, instructions=""):
        super().__init__(instructions=instructions)

# ----------------------------
# Logging helpers
# ----------------------------
def log_to_file(room_name, role, message):
    os.makedirs("chat_logs", exist_ok=True)
    file_path = os.path.join("chat_logs", f"{room_name}.txt")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{role}: {message}\n")

# ----------------------------
# Replace {{language}}
# ----------------------------
def replace_language(obj, target_language):
    if isinstance(obj, dict):
        return {k: replace_language(v, target_language) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_language(v, target_language) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\{\{\s*language\s*\}\}", target_language, obj)
    else:
        return obj

# ----------------------------
# Entrypoint
# ----------------------------
async def entrypoint(ctx: agents.JobContext):
    metadata = {}
    if hasattr(ctx.job, 'metadata') and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
            print(f"📦 Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    if metadata.get("source") != "zabano":
        print("⚠️ Non-zabano job, skipping...")
        return

    agent_type = metadata.get("agent_type", "tutor")
    target_language = metadata.get("language", "English")
    config = metadata.get("config", {})

    config = replace_language(config, target_language)

    await ctx.connect()
    await asyncio.sleep(0.5)

    participants = ctx.room.remote_participants
    agent_count = sum(1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    if agent_count > 0:
        print("⚠️ Existing agent in room — skipping startup")
        return

    session_active = True  # flag to prevent logging after shutdown

    try:
        voice_choices = config.get("livekit", {}).get("voice_choices", ["nova"])
        voice = random.choice(voice_choices)

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

        # ----------------------------
        # Participant left handler
        # ----------------------------
        async def handle_user_left(participant):
            nonlocal session_active
            print(f"👋 Participant left: {participant.identity}")
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return

            session_active = False  # prevent logging after shutdown

            try:
                await session.aclose()  # safely stop all agent tasks
            except Exception as e:
                print("❌ Error stopping session:", e)

            await asyncio.sleep(0.1)
            try:
                if ctx.room.isconnected():
                    await ctx.room.disconnect()
                    print("🛑 Room disconnected")
            except Exception as e:
                print("❌ Error disconnecting room:", e)

        ctx.room.on("participant_disconnected", lambda p: asyncio.create_task(handle_user_left(p)))

        # ----------------------------
        # Event handlers
        # ----------------------------
        last_stt_time = 0
        async def on_transcription(text: str):
            nonlocal last_stt_time
            now = time.time()
            if now - last_stt_time > 0.2:  # debounce STT logging
                last_stt_time = now
                if session_active:
                    print("🎙️ STT:", text)

        async def on_llm_output(text: str):
            if session_active:
                print("🤖 LLM:", text)

        session.on("user_input_transcribed", lambda ev: asyncio.create_task(on_transcription(ev.transcript)))

        async def handle_conversation_item(ev):
            if not session_active:
                return
            await asyncio.to_thread(
                log_to_file,
                ctx.room.name,
                "agent" if ev.item.role == "assistant" else "user",
                " ".join(ev.item.content) if isinstance(ev.item.content, list) else ev.item.content
            )

        session.on("conversation_item_added", lambda ev: asyncio.create_task(handle_conversation_item(ev)))

        # ----------------------------
        # Create agent
        # ----------------------------
        behavior = config.get("behavior", {})
        instructions_text = json.dumps(behavior, ensure_ascii=False)
        agent = DynamicAssistant(instructions=instructions_text)

        # Start session with agent
        await session.start(room=ctx.room, agent=agent)
        await asyncio.sleep(0.5)

        # Generate initial greeting / instructions
        await session.generate_reply(instructions=instructions_text)
        print("✅ Agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise

# ----------------------------
# Run CLI
# ----------------------------
if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
