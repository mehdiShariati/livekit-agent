import json
import os
import random
import asyncio
import asyncpg

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero
from livekit.plugins.openai import realtime

load_dotenv(".env")

DB_POOL = None


async def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(
            dsn=os.getenv("POSTGRES_URL"),
            min_size=1,
            max_size=5,
        )


async def log_to_postgres(room_name: str, role: str, message: str):
    await init_db_pool()
    query = """
        INSERT INTO chat_logs (room_name, role, message, created_at)
        VALUES ($1, $2, $3, NOW())
    """
    async with DB_POOL.acquire() as conn:
        await conn.execute(query, room_name, role, message)


def normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def build_instructions(config: dict) -> str:
    behavior = config.get("behavior", {})

    base = """
You are a friendly, natural real-time voice assistant for Zabano.
Speak clearly and briefly.
Be conversational, responsive, and helpful.
If the user speaks Persian, reply in Persian.
If the user speaks English, reply in English.
If the conversation is mixed Persian and English, adapt naturally.
"""

    if not behavior:
        return base.strip()

    if isinstance(behavior, str):
        return f"{base.strip()}\n\nAdditional behavior:\n{behavior}"

    return f"{base.strip()}\n\nBehavior config:\n{json.dumps(behavior, ensure_ascii=False, indent=2)}"


class DynamicAssistant(Agent):
    def __init__(self, instructions: str = ""):
        super().__init__(instructions=instructions)


async def entrypoint(ctx: agents.JobContext):
    metadata = {}

    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
        try:
            metadata = (
                json.loads(ctx.job.metadata)
                if isinstance(ctx.job.metadata, str)
                else ctx.job.metadata
            )
            print(f"📦 Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to parse metadata: {e}")

    if metadata.get("source") != "zabano":
        print("⚠️ Non-zabano job, skipping...")
        return

    config = metadata.get("config", {})

    await ctx.connect()
    await asyncio.sleep(0.5)

    participants = ctx.room.remote_participants
    agent_count = sum(
        1
        for p in participants.values()
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )

    if agent_count > 0:
        print("⚠️ Existing agent in room — skipping startup")
        return

    try:
        voice_choices = config.get("livekit", {}).get("voice_choices", ["nova"])
        voice = random.choice(voice_choices)

        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                kwargs["task"] = "transcribe"
                kwargs.pop("translate", None)
                return await super().transcribe(*args, **kwargs)

        instructions_text = build_instructions(config)
        agent = DynamicAssistant(instructions=instructions_text)

        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=realtime.RealtimeModel(
                model="gpt-realtime-1.5",
            ),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        async def handle_user_left(participant):
            print(f"👋 Participant left: {participant.identity}")

            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return

            try:
                await session.aclose()
            except Exception as e:
                print(f"Error closing session: {e}")

            try:
                await ctx.room.disconnect()
            except Exception as e:
                print(f"Error disconnecting room: {e}")

        ctx.room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_user_left(p)),
        )

        async def on_transcription(text: str):
            print("🎙️ STT:", text)

        session.on(
            "user_input_transcribed",
            lambda ev: asyncio.create_task(on_transcription(ev.transcript)),
        )

        session.on(
            "conversation_item_added",
            lambda ev: asyncio.create_task(
                log_to_postgres(
                    ctx.room.name,
                    "assistant" if getattr(ev.item, "role", "") == "assistant" else "user",
                    normalize_content(getattr(ev.item, "content", "")),
                )
            ),
        )

        await session.start(room=ctx.room, agent=agent)
        await asyncio.sleep(0.5)

        print("✅ Agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)