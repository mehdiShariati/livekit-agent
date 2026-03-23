import asyncio
import contextlib
import hashlib
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


def stable_lock_key(room_name: str) -> int:
    digest = hashlib.sha256(room_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


class JobConfig(BaseModel):
    native_language: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    speaking_language: str = ""
    assessment_type: str = ""
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
            CREATE TABLE IF NOT EXISTS agent_jobs (
                room_name TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                transcript_room_name TEXT NOT NULL,
                dispatch_id TEXT,
                status TEXT NOT NULL,
                metadata_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)


@app.on_event("startup")
async def startup_event():
    global worker_server

    await init_db_pool()
    await init_tables()

    worker_server = AgentServer()
    worker_server.rtc_session(
        entrypoint,
        agent_name="zabano_agent",
    )

    asyncio.create_task(worker_server.run())


@app.on_event("shutdown")
async def shutdown_event():
    global DB_POOL, worker_server

    if worker_server is not None:
        with contextlib.suppress(Exception):
            await worker_server.aclose()

    if DB_POOL is not None:
        with contextlib.suppress(Exception):
            await DB_POOL.close()


def build_dispatch_metadata(request: JobRequest) -> dict[str, Any]:
    transcript_room_name = (request.transcript_room_name or request.room_name).strip()

    return {
        "source": "zabano",
        "agent_type": request.agent_type.strip() or "tutor",
        "transcript_room_name": transcript_room_name,
        "config": request.config.model_dump(),
    }


def create_livekit_api() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )


@app.post("/jobs")
async def create_job(request: JobRequest):
    now = datetime.now(timezone.utc)
    transcript_room_name = (request.transcript_room_name or request.room_name).strip()
    metadata_dict = build_dispatch_metadata(request)
    lock_key = stable_lock_key(request.room_name)

    async with DB_POOL.acquire() as conn:
        async with conn.transaction():
            lock_acquired = await conn.fetchval(
                "SELECT pg_try_advisory_xact_lock($1::bigint)",
                lock_key,
            )
            if not lock_acquired:
                return {
                    "status": "already_active",
                    "room": request.room_name,
                    "agent_type": request.agent_type,
                    "message": "A job creation is already in progress for this room.",
                }

            row = await conn.fetchrow(
                "SELECT room_name, status, dispatch_id FROM agent_jobs WHERE room_name = $1",
                request.room_name,
            )

            if row and row["status"] in ("dispatching", "pending", "running"):
                return {
                    "status": "already_active",
                    "room": request.room_name,
                    "agent_type": request.agent_type,
                    "message": "An active agent job already exists for this room.",
                }

            await conn.execute("""
                INSERT INTO agent_jobs (
                    room_name, agent_type, transcript_room_name, dispatch_id,
                    status, metadata_json, created_at, updated_at
                )
                VALUES ($1, $2, $3, NULL, 'dispatching', $4::jsonb, $5, $5)
                ON CONFLICT (room_name)
                DO UPDATE SET
                    agent_type = EXCLUDED.agent_type,
                    transcript_room_name = EXCLUDED.transcript_room_name,
                    dispatch_id = NULL,
                    status = 'dispatching',
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
                SET dispatch_id = $2, status = 'running', updated_at = $3
                WHERE room_name = $1
            """, request.room_name, dispatch.id, datetime.now(timezone.utc))

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
                SET status = 'failed', updated_at = $2
                WHERE room_name = $1
            """, request.room_name, datetime.now(timezone.utc))

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if lkapi is not None:
            await lkapi.aclose()


@app.get("/jobs")
async def list_jobs():
    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch("""
            SELECT room_name, agent_type, transcript_room_name, dispatch_id, status, created_at, updated_at
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
            SELECT room_name, agent_type, transcript_room_name, dispatch_id, status, created_at, updated_at
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
            SET status = 'closing', updated_at = $2
            WHERE room_name = $1
        """, room_name, datetime.now(timezone.utc))

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
            WHERE status IN ('dispatching', 'pending', 'running')
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
        ORDER BY created_at ASC
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