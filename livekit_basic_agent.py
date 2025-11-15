import json
import os
import re
import asyncio
import random
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv(".env")


# ---------------------------------------------
# 👩‍🏫 Dynamic Assistant
# ---------------------------------------------
class DynamicAssistant(Agent):
    def __init__(self, instructions=""):
        super().__init__(instructions=instructions)


# ---------------------------------------------
# Logging helper
# ---------------------------------------------
def log_to_file(room_name, role, message):
    os.makedirs("chat_logs", exist_ok=True)
    file_path = os.path.join("chat_logs", f"{room_name}.txt")
    formatted_message = f"{role}: {message}\n"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(formatted_message)


# ---------------------------------------------
# Helper: Replace {{language}} recursively
# ---------------------------------------------
def replace_language(obj, target_language):
    if isinstance(obj, dict):
        return {k: replace_language(v, target_language) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_language(v, target_language) for v in obj]
    elif isinstance(obj, str):
        return re.sub(r"\{\{\s*language\s*\}\}", target_language, obj)
    else:
        return obj


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

    # Default to tutor if not zabano
    if metadata.get("source") != "zabano":
        print("⚠️ Non-zabano job, skipping...")
        return

    # Extract agent type and language
    agent_type = metadata.get("agent_type", "tutor")
    target_language = metadata.get("language", "English")
    config = metadata.get("config", {})

    # Replace all {{language}} placeholders in config
    config = replace_language(config, target_language)

    # Connect to room
    await ctx.connect()
    await asyncio.sleep(0.5)

    # Avoid multiple agents in same room
    participants = ctx.room.remote_participants
    agent_count = sum(1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    if agent_count > 0:
        print("⚠️ Existing agent in room — skipping startup")
        return

    try:
        # Select voice randomly
        voice_choices = config.get("livekit", {}).get("voice_choices", ["nova"])
        voice = random.choice(voice_choices)

        # Custom Whisper STT
        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                kwargs["task"] = "transcribe"
                kwargs.pop("translate", False)
                return await super().transcribe(*args, **kwargs)

        # Create agent session
        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # Cleanup on user leave
        async def handle_user_left(participant):
            print(f"👋 Participant left: {participant.identity}")
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return
            try:
                await session.close()
            except Exception as e:
                print("Error closing session:", e)
            try:
                await ctx.room.disconnect()
            except Exception as e:
                print("Error disconnecting room:", e)

        ctx.room.on("participant_disconnected", lambda p: asyncio.create_task(handle_user_left(p)))

        # Logging
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

        # Prepare instructions/greeting
        behavior = config.get("behavior", {})
        if behavior:
            instructions_text = f"Target language: {target_language}\n{json.dumps(behavior, ensure_ascii=False)}"
        else:
            instructions_text = f"Hello! Let's start your {target_language} session."

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
