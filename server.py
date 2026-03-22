import asyncio
import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from livekit import api
from livekit.agents import AgentServer

from livekit_basic_agent import entrypoint

load_dotenv(".env")

app = FastAPI(title="LiveKit Agent Manager")

DB_POOL = None
DB_POOL_LOCK = asyncio.Lock()
worker_server = None


# ============================================================
# Request models
# ============================================================

class JobConfig(BaseModel):
    native_language: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    learner_level: str = Field(..., min_length=1)

    lesson_topic: str = Field(..., min_length=1)
    lesson_goal: str = Field(..., min_length=1)

    learner_interests_text: str = ""
    learner_strengths_text: str = ""
    learner_weaknesses_text: str = ""

    progress_summary: str = ""
    memory_summary: str = ""
    resume_instruction: str = ""
    opening_line: str = ""


class JobRequest(BaseModel):
    room_name: str = Field(..., min_length=1)
    agent_type: str = Field(default="tutor", min_length=1)
    transcript_room_name: str | None = None
    config: JobConfig


# ============================================================
# DB helpers
# ============================================================

async def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        async with DB_POOL_LOCK:
            if DB_POOL is None:
                DB_POOL = await asyncpg.create_pool(
                    dsn=os.getenv("POSTGRES_URL"),
                    min_size=1,
                    max_size=10,
                )


async def init_tables():
    async with DB_POOL.acquire() as conn:
        await conn.execute("""
                           CREATE TABLE IF NOT EXISTS agent_jobs
                           (
                               room_name
                               TEXT
                               PRIMARY
                               KEY,
                               agent_type
                               TEXT
                               NOT
                               NULL,
                               transcript_room_name
                               TEXT
                               NOT
                               NULL,
                               dispatch_id
                               TEXT,
                               status
                               TEXT
                               NOT
                               NULL,
                               metadata_json
                               JSONB
                               NOT
                               NULL,
                               created_at
                               TIMESTAMPTZ
                               NOT
                               NULL,
                               updated_at
                               TIMESTAMPTZ
                               NOT
                               NULL
                           )
                           """)


# ============================================================
# Startup / shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():
    global worker_server

    print("🚀 Starting application...")

    await init_db_pool()
    await init_tables()
    print("✅ Database pool initialized and tables ensured")

    worker_server = AgentServer()
    worker_server.rtc_session(
        entrypoint,
        agent_name="zabano_agent",
    )

    asyncio.create_task(worker_server.run())
    print("✅ LiveKit worker started")


@app.on_event("shutdown")
async def shutdown_event():
    global DB_POOL, worker_server

    if worker_server is not None:
        try:
            await worker_server.aclose()
            print("✅ LiveKit worker closed")
        except Exception as e:
            print(f"⚠️ Error closing LiveKit worker: {e}")

    if DB_POOL is not None:
        try:
            await DB_POOL.close()
            print("✅ Database pool closed")
        except Exception as e:
            print(f"⚠️ Error closing DB pool: {e}")


# ============================================================
# Metadata builder
# ============================================================

def build_dispatch_metadata(request: JobRequest) -> dict[str, Any]:
    transcript_room_name = (request.transcript_room_name or request.room_name).strip()

    return {
        "source": "zabano",
        "agent_type": request.agent_type.strip() or "tutor",
        "transcript_room_name": transcript_room_name,
        "config": request.config.model_dump(),
    }


# ============================================================
# LiveKit API helper
# ============================================================

def create_livekit_api() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )


# ============================================================
# Routes
# ============================================================

@app.post("/jobs")
async def create_job(request: JobRequest):
    now = datetime.now(timezone.utc)
    transcript_room_name = (request.transcript_room_name or request.room_name).strip()
    metadata_dict = build_dispatch_metadata(request)

    print(f"🔔 Received job request room={request.room_name} agent_type={request.agent_type}")

    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT room_name, status, dispatch_id FROM agent_jobs WHERE room_name = $1",
            request.room_name,
        )

        if row and row["status"] in ("pending", "running"):
            return {
                "status": "already_active",
                "room": request.room_name,
                "agent_type": request.agent_type,
                "message": "An active agent job already exists for this room.",
            }

        await conn.execute("""
                           INSERT INTO agent_jobs (room_name, agent_type, transcript_room_name, dispatch_id,
                                                   status, metadata_json, created_at, updated_at)
                           VALUES ($1, $2, $3, NULL, 'pending', $4::jsonb, $5, $5) ON CONFLICT (room_name)
            DO
                           UPDATE SET
                               agent_type = EXCLUDED.agent_type,
                               transcript_room_name = EXCLUDED.transcript_room_name,
                               dispatch_id = NULL,
                               status = 'pending',
                               metadata_json = EXCLUDED.metadata_json,
                               updated_at = EXCLUDED.updated_at
                           """,
                           request.room_name,
                           request.agent_type,
                           transcript_room_name,
                           json.dumps(metadata_dict, ensure_ascii=False),
                           now)

    lkapi = None
    try:
        lkapi = create_livekit_api()

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="zabano_agent",
                room=request.room_name,
                metadata=json.dumps(metadata_dict, ensure_ascii=False),
            )
        )

        async with DB_POOL.acquire() as conn:
            await conn.execute("""
                               UPDATE agent_jobs
                               SET dispatch_id = $2,
                                   status      = 'running',
                                   updated_at  = $3
                               WHERE room_name = $1
                               """, request.room_name, dispatch.id, now)

        print(f"✅ Dispatch created successfully room={request.room_name}")

        return {
            "status": "started",
            "room": request.room_name,
            "agent_type": request.agent_type,
            "message": "Agent dispatch created successfully.",
        }

    except Exception as e:
        async with DB_POOL.acquire() as conn:
            await conn.execute("""
                               UPDATE agent_jobs
                               SET status     = 'failed',
                                   updated_at = $2
                               WHERE room_name = $1
                               """, request.room_name, datetime.now(timezone.utc))

        print(f"❌ Dispatch creation failed room={request.room_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if lkapi is not None:
            await lkapi.aclose()


@app.get("/jobs")
async def list_jobs():
    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch("""
                                SELECT room_name,
                                       agent_type,
                                       transcript_room_name,
                                       dispatch_id,
                                       status,
                                       created_at,
                                       updated_at
                                FROM agent_jobs
                                ORDER BY updated_at DESC
                                """)

    return {
        "count": len(rows),
        "jobs": [dict(row) for row in rows],
    }


@app.get("/jobs/{room_name}")
async def get_job(room_name: str):
    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow("""
                                  SELECT room_name,
                                         agent_type,
                                         transcript_room_name,
                                         dispatch_id,
                                         status,
                                         created_at,
                                         updated_at
                                  FROM agent_jobs
                                  WHERE room_name = $1
                                  """, room_name)

    if not row:
        raise HTTPException(status_code=404, detail="No job found for this room.")

    return dict(row)


@app.delete("/jobs/{room_name}")
async def remove_job(room_name: str):
    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT room_name, status, dispatch_id FROM agent_jobs WHERE room_name = $1",
            room_name,
        )

        if not row:
            return {
                "status": "not_found",
                "room": room_name,
                "message": "No job found.",
            }

        await conn.execute("""
                           UPDATE agent_jobs
                           SET status     = 'closing',
                               updated_at = $2
                           WHERE room_name = $1
                           """, room_name, datetime.now(timezone.utc))

    # Note:
    # This marks the job as closing in your DB.
    # If you later add explicit LiveKit-side session termination, do it here too.
    return {
        "status": "closing",
        "room": room_name,
        "message": "Job marked as closing.",
    }


@app.get("/health")
async def health_check():
    async with DB_POOL.acquire() as conn:
        active_count = await conn.fetchval("""
                                           SELECT COUNT(*)
                                           FROM agent_jobs
                                           WHERE status IN ('pending', 'running')
                                           """)

    return {
        "status": "healthy",
        "active_jobs": active_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/logs/{room_name}")
async def get_chat_log(room_name: str):
    global DB_POOL

    if DB_POOL is None:
        raise HTTPException(status_code=500, detail="Database pool is not initialized")

    query = """
            SELECT role, message, created_at
            FROM chat_logs
            WHERE room_name = $1
            ORDER BY created_at ASC \
            """

    async with DB_POOL.acquire() as conn:
        records = await conn.fetch(query, room_name)

    if not records:
        raise HTTPException(status_code=404, detail=f"No logs found for room '{room_name}'")

    content = "\n".join(f"{r['role']}: {r['message']}" for r in records)
    buffer = io.BytesIO(content.encode("utf-8"))

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={room_name}.txt"},
    )
