import json
import os
import asyncio
import io
import asyncpg

from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from livekit import api
from livekit.agents import AgentServer

from livekit_basic_agent import entrypoint

load_dotenv(".env")

app = FastAPI(title="LiveKit Agent Manager")

DB_POOL = None
DB_POOL_LOCK = asyncio.Lock()

active_dispatches = {}
dispatch_locks = {}
worker_server = None


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


async def get_room_lock(room_name: str):
    if room_name not in dispatch_locks:
        dispatch_locks[room_name] = asyncio.Lock()
    return dispatch_locks[room_name]


class JobRequest(BaseModel):
    room_name: str
    agent_type: str = "tutor"
    config: dict | None = None


@app.on_event("startup")
async def startup_event():
    global worker_server

    print("🚀 Starting application...")

    await init_db_pool()
    print("✅ Database pool initialized")

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


@app.post("/jobs")
async def create_job(request: JobRequest):
    timestamp = datetime.now().isoformat()
    print(f"🔔 [{timestamp}] Received job request:")
    print(f"   Room: {request.room_name}")
    print(f"   Agent Type: {request.agent_type}")
    print(f"   Config: {request.config}")

    lock = await get_room_lock(request.room_name)

    async with lock:
        lkapi = None
        try:
            lkapi = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )

            metadata_dict = {
                "agent_type": request.agent_type,
                "source": "zabano",
            }
            if request.config:
                metadata_dict["config"] = request.config

            dispatch = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="zabano_agent",
                    room=request.room_name,
                    metadata=json.dumps(metadata_dict),
                )
            )

            active_dispatches[request.room_name] = {
                "agent_type": request.agent_type,
                "dispatch_id": dispatch.id,
                "timestamp": timestamp,
                "config": request.config,
            }

            print(f"✅ [{timestamp}] Dispatch created successfully for room {request.room_name}")
            return {
                "status": "started",
                "agent_type": request.agent_type,
                "room": request.room_name,
                "dispatch_id": dispatch.id,
                "message": f"Agent {request.agent_type} started in room {request.room_name}",
            }

        except Exception as e:
            print(f"❌ [{timestamp}] Dispatch error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if lkapi is not None:
                await lkapi.aclose()


@app.delete("/jobs/{room_name}")
async def remove_job(room_name: str):
    timestamp = datetime.now().isoformat()

    if room_name in active_dispatches:
        dispatch_info = active_dispatches.pop(room_name)

        print(f"🗑️ [{timestamp}] Removed dispatch tracking for room: {room_name}")
        return {
            "status": "removed",
            "room": room_name,
            "dispatch_info": dispatch_info,
        }

    print(f"⚠️ [{timestamp}] No dispatch found for room: {room_name}")
    return {
        "status": "not_found",
        "room": room_name,
        "message": f"No active dispatch found for room {room_name}",
    }


@app.get("/jobs")
async def list_jobs():
    return {
        "active_dispatches": active_dispatches,
        "count": len(active_dispatches),
        "rooms": list(active_dispatches.keys()),
    }


@app.get("/jobs/{room_name}")
async def get_job(room_name: str):
    if room_name in active_dispatches:
        return {
            "status": "active",
            "room": room_name,
            "dispatch": active_dispatches[room_name],
        }

    raise HTTPException(status_code=404, detail=f"No active dispatch found for room {room_name}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_dispatches": len(active_dispatches),
        "timestamp": datetime.now().isoformat(),
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