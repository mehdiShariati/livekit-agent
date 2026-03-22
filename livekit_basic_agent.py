import json
import os
import random
import asyncio
import asyncpg

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

DB_POOL = None
DB_POOL_LOCK = asyncio.Lock()


async def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        async with DB_POOL_LOCK:
            if DB_POOL is None:
                DB_POOL = await asyncpg.create_pool(
                    dsn=os.getenv("POSTGRES_URL"),
                    min_size=1,
                    max_size=5,
                )


async def log_to_postgres(room_name: str, role: str, message: str):
    try:
        await init_db_pool()
        query = """
            INSERT INTO chat_logs (room_name, role, message, created_at)
            VALUES ($1, $2, $3, NOW())
        """
        async with DB_POOL.acquire() as conn:
            await conn.execute(query, room_name, role, message)
    except Exception as e:
        print(f"DB log error: {e}")


def normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def build_instructions(agent_type: str, config_override: dict | None = None) -> str:
    base_map = {
        "onboarding": """
You are a friendly onboarding guide for zabano.com.
Speak in Persian.
Keep responses short, warm, and motivating.
""".strip(),
        "assessment": """
You are an English proficiency assessor.
Conduct a short conversation to evaluate the user's English speaking and comprehension.
Ask open questions and keep ratings private.
Speak partly in English and partly in Persian.
""".strip(),
        "tutor": """
You are an expert English tutor for Persian speakers.
Explain when needed in Persian, but keep the conversation natural.
Be kind, interactive, patient, and brief.
""".strip(),
    }

    instructions = base_map.get(agent_type, base_map["tutor"])

    if not config_override:
        return instructions

    behavior = config_override.get("behavior", {})
    if not behavior:
        return instructions

    tone = behavior.get("tone")
    rules = behavior.get("rules", {})
    user_level = rules.get("user_level")
    topic_focus = rules.get("topic_focus", {})
    topic = topic_focus.get("topic")
    topic_description = topic_focus.get("description")
    conciseness = rules.get("conciseness")
    interaction_style = rules.get("interaction_style")

    extra_lines = []

    if tone:
        extra_lines.append(f"Tone: {tone}")
    if user_level:
        extra_lines.append(f"User level: {user_level}")
    if conciseness:
        extra_lines.append(conciseness)
    if topic:
        extra_lines.append(f"Topic focus: {topic}")
    if topic_description:
        extra_lines.append(topic_description)
    if interaction_style:
        extra_lines.append(interaction_style)

    if extra_lines:
        instructions += "\n\nAdditional behavior:\n" + "\n".join(f"- {line}" for line in extra_lines)

    return instructions


class DynamicAssistant(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


async def entrypoint(ctx: agents.JobContext):
    metadata = {}

    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
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
            print(f"⚠️ Ignoring non-zabano job: {metadata}")
            return
    else:
        agent_type = metadata.get("agent_type", "tutor")

    config_override = metadata.get("config", {}) or {}

    await ctx.connect()
    await asyncio.sleep(0.5)

    print(f"✅ Connected to room: {ctx.room.name}")
    print(f"👥 Remote participants: {list(ctx.room.remote_participants.keys())}")

    participants = ctx.room.remote_participants
    agent_count = sum(
        1
        for participant in participants.values()
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )

    if agent_count > 0:
        print(f"⚠️ {agent_count} agent(s) already in room {ctx.room.name}, skipping")
        return

    try:
        voice_choices = config_override.get("livekit", {}).get("voice_choices", ["marin", "nova"])
        voice = random.choice(voice_choices)

        instructions_text = build_instructions(agent_type, config_override)
        agent = DynamicAssistant(instructions=instructions_text)

        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-1.5"),
                voice=voice,
            ),
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

        session.on(
            "user_input_transcribed",
            lambda ev: print(f"🎙️ STT: {ev.transcript}"),
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

        print(f"🚀 Starting {agent_type} agent in room {ctx.room.name} with voice {voice}")
        await session.start(room=ctx.room, agent=agent)

        greeting_map = {
            "onboarding": "سلام! به زبانو خوش آمدید. چطور می‌تونم کمکتون کنم؟",
            "assessment": "Hello! سلام! Ready to test your English?",
            "tutor": "سلام! من معلم انگلیسی شما هستم. بیایید شروع کنیم!",
        }
        greeting = greeting_map.get(agent_type, greeting_map["tutor"])

        await session.generate_reply(instructions=greeting)

        print(f"✅ {agent_type} agent started successfully")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)