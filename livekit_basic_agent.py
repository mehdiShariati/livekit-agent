#!/usr/bin/env python3
import os
import json
import re
import time
import asyncio
import random
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv(".env")

# ----------------------------------------
# CONSTANTS
# ----------------------------------------
LLM_CONCURRENCY = 4
FILE_QUEUE_MAXSIZE = 2000
STT_DEBOUNCE_SECONDS = 0.25
LOG_FOLDER = "chat_logs"

ROOM_STARTUP_LOCKS = {}
LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY)


# ----------------------------------------
# SIMPLE AGENT CLASS
# ----------------------------------------
class DynamicAssistant(Agent):
    def __init__(self, instructions: str = ""):
        super().__init__(instructions=instructions)


# ----------------------------------------
# PER-ROOM FILE LOGGER
# ----------------------------------------
class FileLogger:
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.queue = asyncio.Queue(maxsize=FILE_QUEUE_MAXSIZE)
        self.running = True
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        os.makedirs(LOG_FOLDER, exist_ok=True)
        while self.running or not self.queue.empty():
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            role, message, ts = item
            date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            path_dir = os.path.join(LOG_FOLDER, self.room_name)
            os.makedirs(path_dir, exist_ok=True)
            path = os.path.join(path_dir, f"{date_str}.txt")

            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.utcfromtimestamp(ts).isoformat()}Z] {role}: {message}\n")
            except Exception as e:
                print("❌ log write error:", e)

            self.queue.task_done()

    async def log(self, role: str, message: str):
        if not message:
            return
        try:
            self.queue.put_nowait((role, message, time.time()))
        except asyncio.QueueFull:
            print(f"⚠️ Log queue full for room {self.room_name}, dropping.")

    async def close(self):
        self.running = False
        try:
            await asyncio.wait_for(self.queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            print("⚠️ timeout flushing logs")

        if not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except:
                pass


# ----------------------------------------
# SAFE LLM CALL
# ----------------------------------------
async def safe_generate_reply(session: AgentSession, instructions: str):
    async with LLM_SEMAPHORE:
        return await session.generate_reply(instructions=instructions)


# ----------------------------------------
# ENTRYPOINT — CALLED BY server.py DISPATCH
# ----------------------------------------
async def entrypoint(ctx: agents.JobContext):
    # Parse metadata
    metadata = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except:
            pass

    # Only run for your system
    if metadata.get("source") != "zabano":
        print("⚠️ Skipping non-zabano job")
        return

    room_name = ctx.room.name

    # Make sure only ONE agent per room starts
    room_lock = ROOM_STARTUP_LOCKS.setdefault(room_name, asyncio.Lock())
    async with room_lock:
        # Avoid spawning duplicate agents
        agent_count = sum(
            1 for p in ctx.room.remote_participants.values()
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        )
        if agent_count > 0:
            print(f"[{room_name}] ⚠️ Agent already exists, skip")
            return

        print(f"[{room_name}] 🚀 Starting agent...")

        # Setup logger
        logger = FileLogger(room_name)
        session_running = True

        # Choose random voice
        voice_choices = (metadata.get("config") or {}).get("livekit", {}).get("voice_choices", ["nova"])
        voice = random.choice(voice_choices)

        # Custom STT (no translate)
        class CustomWhisper(openai.STT):
            async def transcribe(self, *args, **kwargs):
                kwargs["task"] = "transcribe"
                kwargs["translate"] = False
                return await super().transcribe(*args, **kwargs)

        session = AgentSession(
            stt=CustomWhisper(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        last_stt = 0

        # ----------------------
        # EVENT HANDLERS
        # ----------------------
        async def on_stt(text: str):
            nonlocal last_stt, session_running

            if not session_running:
                return

            if not text.strip():
                return

            now = time.time()
            if now - last_stt < STT_DEBOUNCE_SECONDS:
                return
            last_stt = now

            await logger.log("stt", text)
            print(f"[{room_name}] 🎙️ STT: {text}")

        async def on_item(ev):
            if not session_running:
                return

            content = ev.item.content
            if isinstance(content, list):
                content = " ".join(content)

            if getattr(ev.item, "final", True) is False:
                return

            role = "agent" if ev.item.role in ("assistant", "agent") else "user"
            await logger.log(role, content)

        session.on("user_input_transcribed", lambda ev: asyncio.create_task(on_stt(ev.transcript)))
        session.on("conversation_item_added", lambda ev: asyncio.create_task(on_item(ev)))

        # ----------------------
        # SHUTDOWN HANDLER
        # ----------------------
        async def shutdown(reason="left"):
            nonlocal session_running
            if not session_running:
                return
            session_running = False

            print(f"[{room_name}] 🔻 Shutdown: {reason}")

            try:
                await session.aclose()
            except Exception as e:
                print("close error:", e)

            try:
                await logger.close()
            except Exception as e:
                print("log close error:", e)

            try:
                if ctx.room.isconnected():
                    await ctx.room.disconnect()
            except:
                pass

        ctx.room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(
                shutdown("participant disconnected") if p.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT else None
            )
        )

        # ----------------------
        # START SESSION
        # ----------------------
        try:
            behavior = (metadata.get("config") or {}).get("behavior", {})
            system_prompt = behavior.get("system", "You are a helpful tutor.")

            agent = DynamicAssistant(system_prompt)
            await session.start(room=ctx.room, agent=agent)

            if not session_running:
                await shutdown("aborted")
                return

            # Greeting
            await safe_generate_reply(session, "greet")

            print(f"[{room_name}] ✅ Agent ready.")

            while session_running:
                await asyncio.sleep(0.5)

        except Exception as e:
            print(f"[{room_name}] ❌ startup error:", e)
            await shutdown("error")
            raise
