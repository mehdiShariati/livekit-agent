import json
import os
import re
import random
import asyncio
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

def log_to_file(room, role, message):
    os.makedirs("chat_logs", exist_ok=True)
    with open(f"chat_logs/{room}.txt", "a", encoding="utf-8") as f:
        f.write(f"{role}: {message}\n")


def replace_language(obj, target_language):
    if isinstance(obj, dict):
        return {k: replace_language(v, target_language) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_language(i, target_language) for i in obj]
    if isinstance(obj, str):
        return re.sub(r"\{\{\s*language\s*\}\}", target_language, obj)
    return obj


class DynamicAssistant(Agent):
    pass


async def entrypoint(ctx: agents.JobContext):

    meta = {}
    if ctx.job.metadata:
        try:
            meta = json.loads(ctx.job.metadata)
        except:
            pass

    if meta.get("source") != "zabano":
        return

    agent_type = meta.get("agent_type", "tutor")
    target_language = meta.get("language", "English")
    config = replace_language(meta.get("config", {}), target_language)

    await ctx.connect()

    # Do not start if another agent is already connected
    if any(p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
           for p in ctx.room.remote_participants.values()):
        print("Agent already running — skipping")
        return

    # Select voice
    voice = random.choice(config.get("livekit", {}).get("voice_choices", ["nova"]))

    class CustomWhisperSTT(openai.STT):
        async def transcribe(self, *a, **kw):
            kw["task"] = "transcribe"
            kw.pop("translate", None)
            return await super().transcribe(*a, **kw)

    session = AgentSession(
        stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
        tts=openai.TTS(voice=voice),
        vad=silero.VAD.load(),
    )

    async def on_left(p):
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return
        await session.close()
        await asyncio.sleep(0.25)
        await ctx.room.disconnect()

    ctx.room.on("participant_disconnected",
        lambda p: asyncio.create_task(on_left(p))
    )

    session.on(
        "conversation_item_added",
        lambda ev: log_to_file(
            ctx.room.name,
            "assistant" if ev.item.role == "assistant" else "user",
            " ".join(ev.item.content) if isinstance(ev.item.content, list) else ev.item.content
        )
    )

    instructions = json.dumps(config.get("behavior", {}), ensure_ascii=False)
    agent = DynamicAssistant(instructions=instructions)

    # ✅ FIXED — only pass the agent
    await session.start(agent=agent)

    await asyncio.sleep(0.25)
    await session.generate_reply(instructions=instructions)

    print("Agent started ✔️")


if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
