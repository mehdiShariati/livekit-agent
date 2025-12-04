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

# ----------------------------
# Configurable constants
# ----------------------------
LLM_CONCURRENCY = 4              # number of simultaneous LLM requests across all rooms
FILE_QUEUE_MAXSIZE = 2000        # per-room file write queue size
STT_DEBOUNCE_SECONDS = 0.25     # ignore extremely frequent STT callbacks
LOG_FOLDER = "chat_logs"
ROOM_STARTUP_LOCKS = {}         # per-room asyncio.Lock to prevent race-starts (per-process)
LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY)

# ----------------------------
# Minimal DynamicAssistant
# ----------------------------
class DynamicAssistant(Agent):
    def __init__(self, instructions: str = ""):
        super().__init__(instructions=instructions)

# ----------------------------
# Simple per-room file logger (keeps concerns narrow)
# ----------------------------
class FileLogger:
    """
    Per-room file logger. Create one FileLogger per active room.
    - Has its own asyncio.Queue to avoid cross-room bottlenecks.
    - Writer task flushes messages to a daily file.
    - Stop() waits for pending items to be written then cancels task.
    """
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.queue = asyncio.Queue(maxsize=FILE_QUEUE_MAXSIZE)
        self._running = True
        self._task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self):
        os.makedirs(LOG_FOLDER, exist_ok=True)
        try:
            while self._running or not self.queue.empty():
                try:
                    role, message, ts = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                file_dir = os.path.join(LOG_FOLDER, self.room_name)
                os.makedirs(file_dir, exist_ok=True)
                file_path = os.path.join(file_dir, f"{date_str}.txt")
                try:
                    # append with timestamp
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.utcfromtimestamp(ts).isoformat()}Z] {role}: {message}\n")
                except Exception as e:
                    print("❌ File write error:", e)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            # graceful cancellation path
            pass

    async def log(self, role: str, message: str):
        """Enqueue. If queue is full, drop the message with a short warning to stdout."""
        if not message:
            return
        try:
            self.queue.put_nowait((role, message, time.time()))
        except asyncio.QueueFull:
            print(f"⚠️ Log queue full for {self.room_name} — dropping log.")

    async def stop(self):
        """Stop accepting new logs, wait for queue drained, then cancel task."""
        self._running = False
        try:
            # wait up to 5s for queue to flush
            await asyncio.wait_for(self.queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout waiting for logs to flush for room {self.room_name}")
        # cancel the writer task
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

# ----------------------------
# Utilities
# ----------------------------
def replace_language(obj, target_language: str):
    """Simple replace for '{{language}}' tokens. Keep it lightweight."""
    if isinstance(obj, dict):
        return {k: replace_language(v, target_language) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_language(v, target_language) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"\{\{\s*language\s*\}\}", target_language, obj)
    return obj

async def safe_generate_reply(session: AgentSession, instructions: str):
    """Limit concurrent LLM calls with a semaphore."""
    async with LLM_SEMAPHORE:
        return await session.generate_reply(instructions=instructions)

# ----------------------------
# Entrypoint
# ----------------------------
async def entrypoint(ctx: agents.JobContext):
    # parse metadata safely
    metadata = {}
    if hasattr(ctx.job, "metadata") and ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata) if isinstance(ctx.job.metadata, str) else ctx.job.metadata
        except Exception as e:
            print("❌ Failed to parse metadata:", e)

    if metadata.get("source") != "zabano":
        print("⚠️ Non-zabano job, skipping...")
        return

    room_name = ctx.room.name
    # ensure one startup operation per room per process
    room_lock = ROOM_STARTUP_LOCKS.setdefault(room_name, asyncio.Lock())

    async with room_lock:
        # check again to avoid race within process
        participants = ctx.room.remote_participants
        agent_count = sum(1 for p in participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
        if agent_count > 0:
            print("⚠️ Existing agent in room — skipping startup")
            return

        # Setup logger for this room
        file_logger = FileLogger(room_name)

        # flag to guard further actions after shutdown
        session_active = True

        # create session components
        voice_choices = (metadata.get("config") or {}).get("livekit", {}).get("voice_choices", ["nova"])
        voice = random.choice(voice_choices)

        class CustomWhisperSTT(openai.STT):
            async def transcribe(self, *args, **kwargs):
                # explicitly enforce transcription (no automatic translation)
                kwargs["task"] = "transcribe"
                kwargs["translate"] = False
                return await super().transcribe(*args, **kwargs)

        # build a session
        session = AgentSession(
            stt=CustomWhisperSTT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
            tts=openai.TTS(voice=voice),
            vad=silero.VAD.load(),
        )

        # ----------------------------
        # event handlers
        # ----------------------------
        last_stt_time = 0.0

        async def on_transcription(text: str):
            nonlocal last_stt_time, session_active
            if not session_active:
                return
            if not text or not text.strip():
                return
            now = time.time()
            if now - last_stt_time < STT_DEBOUNCE_SECONDS:
                return
            last_stt_time = now
            # log STT to console and file
            print(f"[{room_name}] 🎙️ STT: {text}")
            await file_logger.log("stt", text)

        async def on_conversation_item(ev):
            nonlocal session_active
            if not session_active:
                return
            # support either list or single content
            content = ev.item.content
            if isinstance(content, list):
                content = " ".join(content)
            # avoid logging partial updates if possible (best effort)
            is_final = getattr(ev.item, "final", None)
            if is_final is False:
                return
            role = "agent" if getattr(ev.item, "role", "") in ("assistant", "agent") else "user"
            await file_logger.log(role, content)

        # attach handlers
        session.on("user_input_transcribed", lambda ev: asyncio.create_task(on_transcription(ev.transcript)))
        session.on("conversation_item_added", lambda ev: asyncio.create_task(on_conversation_item(ev)))

        # ----------------------------
        # participant-left handler (graceful shutdown)
        # ----------------------------
        async def _shutdown(reason: str = "participant_left"):
            nonlocal session_active
            if not session_active:
                return
            session_active = False
            print(f"[{room_name}] 🔻 Shutdown started: {reason}")
            try:
                await session.aclose()
            except Exception as e:
                print(f"[{room_name}] ❌ Error closing session: {e}")
            # flush logs and stop logger
            try:
                await file_logger.stop()
            except Exception as e:
                print(f"[{room_name}] ❌ Error stopping logger: {e}")
            # attempt room disconnect
            try:
                if ctx.room.isconnected():
                    await ctx.room.disconnect()
                    print(f"[{room_name}] 🛑 Room disconnected")
            except Exception as e:
                print(f"[{room_name}] ❌ Error disconnecting room: {e}")

        # handle participant disconnected events (only non-agent participants trigger agent shutdown)
        async def participant_left_cb(participant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return
            await _shutdown("participant_disconnected")

        ctx.room.on("participant_disconnected", lambda p: asyncio.create_task(participant_left_cb(p)))

        # ----------------------------
        # Create agent & start session
        # ----------------------------
        try:
            behavior = (metadata.get("config") or {}).get("behavior", {})
            # pick a small concise system instruction; don't pass giant JSON dumps to LLM
            system_prompt = behavior.get("system", json.dumps(behavior)) if behavior else "You are a helpful tutor."

            agent = DynamicAssistant(instructions=system_prompt)
            await session.start(room=ctx.room, agent=agent)

            # if someone left during startup, abort cleanly
            if not session_active:
                await _shutdown("aborted_after_start")
                return

            # initial greeting - small simple instruction
            await safe_generate_reply(session, "greet")
            print(f"[{room_name}] ✅ Agent started successfully")

            # keep the entrypoint alive until session is not active
            while session_active:
                await asyncio.sleep(1.0)

        except Exception as e:
            print(f"[{room_name}] ❌ Error starting agent: {e}")
            try:
                await _shutdown("startup_error")
            except Exception:
                pass
            raise

# ----------------------------
# Run CLI
# ----------------------------
if __name__ == "__main__":
    agents.cli.run_app(entrypoint)
