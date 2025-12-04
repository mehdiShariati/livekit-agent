import json
import os
import re
import random
import asyncio
import asyncpg
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")


DB_POOL = None

async def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(
            dsn=os.getenv("POSTGRES_URL"),   # e.g. postgres://user:pass@host/dbname
            min_size=1,
            max_size=5
        )

async def log_to_postgres(room_name, role, message):
    await init_db_pool()
    query = """
        INSERT INTO chat_logs (room_name, role, message, created_at)
        VALUES ($1, $2, $3, NOW())
    """
    async with DB_POOL.acquire() as conn:
        await conn.execute(query, room_name, role, message)


class DynamicAssistant(Agent):
    def __init__(self, instructions=""):
        super().__init__(instructions=instructions)

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

    config = metadata.get("config", {})

    await ctx.connect()
    await asyncio.sleep(0.5)

    participants = ctx.room.remote_participants
    agent_count = sum(1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    if agent_count > 0:
        print("⚠️ Existing agent in room — skipping startup")
        return

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

        async def handle_user_left(participant):
            print(f"👋 Participant left: {participant.identity}")
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return
            try:
                await session.aclose()
            except Exception as e:
                print("Error closing session:", e)
            try:
                await ctx.room.disconnect()
            except Exception as e:
                print("Error disconnecting room:", e)

        ctx.room.on("participant_disconnected", lambda p: asyncio.create_task(handle_user_left(p)))

        async def on_transcription(text: str):
            print("🎙️ STT:", text)

        async def on_llm_output(text: str):
            print("🤖 LLM:", text)

        session.on("user_input_transcribed", lambda ev: asyncio.create_task(on_transcription(ev.transcript)))

        session.on(
            "conversation_item_added",
            lambda ev: asyncio.create_task(
                log_to_postgres(
                    ctx.room.name,
                    "assistant" if ev.item.role == "assistant" else "user",
                    " ".join(ev.item.content) if isinstance(ev.item.content, list) else ev.item.content
                )
            )
        )

        behavior = config.get("behavior", {})
        instructions_text = json.dumps(behavior, ensure_ascii=False)
        agent = DynamicAssistant(instructions=instructions_text)

        await session.start(room=ctx.room, agent=agent)
        await asyncio.sleep(0.5)

        await session.generate_reply(instructions=instructions_text)

        print("✅ Agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
