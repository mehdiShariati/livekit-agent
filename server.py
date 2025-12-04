import json
import os
import asyncio
import asyncpg
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from livekit import api, agents
from livekit_basic_agent import entrypoint
import io

# Load environment variables
load_dotenv(".env")

app = FastAPI(title="LiveKit Agent Manager")

DB_POOL: asyncpg.pool.Pool | None = None

# Track active dispatches per room with details
active_dispatches = {}

# Locks for preventing race conditions per room
dispatch_locks = {}

# Initialize the LiveKit worker
worker = agents.Worker(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="zabano_agent",
        ws_url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
)


async def init_db_pool():
    """Initialize the PostgreSQL connection pool."""
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(
            dsn=os.getenv("POSTGRES_URL"),
            min_size=1,
            max_size=5
        )


async def get_room_lock(room_name: str):
    """Get or create a lock for a specific room."""
    if room_name not in dispatch_locks:
        dispatch_locks[room_name] = asyncio.Lock()
    return dispatch_locks[room_name]


class JobRequest(BaseModel):
    room_name: str
    agent_type: str = "tutor"
    config: dict = None


@app.on_event("startup")
async def startup_event():
    """Startup event: initialize DB pool and start LiveKit worker."""
    print("🚀 Starting application...")

    # Initialize database pool
    await init_db_pool()
    print("✅ Database pool initialized")

    # Start LiveKit worker
    asyncio.create_task(worker.run())
    print("✅ LiveKit worker started")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event: close DB pool."""
    global DB_POOL
    if DB_POOL is not None:
        await DB_POOL.close()
        print("✅ Database pool closed")


@app.post("/jobs")
async def create_job(request: JobRequest):
    """Spawn a new agent in a specific room."""
    timestamp = datetime.now().isoformat()
    print(f"🔔 [{timestamp}] Received job request:")
    print(f"   Room: {request.room_name}")
    print(f"   Agent Type: {request.agent_type}")
    print(f"   Config: {request.config}")

    lock = await get_room_lock(request.room_name)

    async with lock:
        try:
            lkapi = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )

            metadata_dict = {
                "agent_type": request.agent_type,
                "source": "zabano"
            }
            if request.config:
                metadata_dict["config"] = request.config

            dispatch = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="zabano_agent",
                    room=request.room_name,
                    metadata=json.dumps(metadata_dict)
                )
            )

            await lkapi.aclose()

            active_dispatches[request.room_name] = {
                'agent_type': request.agent_type,
                'dispatch_id': dispatch.id,
                'timestamp': timestamp,
                'config': request.config
            }

            print(f"✅ [{timestamp}] Dispatch created successfully for room {request.room_name}")
            return {
                "status": "started",
                "agent_type": request.agent_type,
                "room": request.room_name,
                "dispatch_id": dispatch.id,
                "message": f"Agent {request.agent_type} started in room {request.room_name}"
            }

        except Exception as e:
            print(f"❌ [{timestamp}] Dispatch error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{room_name}")
async def remove_job(room_name: str):
    """Remove tracking for a room."""
    timestamp = datetime.now().isoformat()

    if room_name in active_dispatches:
        dispatch_info = active_dispatches.pop(room_name)

        print(f"🗑️ [{timestamp}] Removed dispatch tracking for room: {room_name}")
        return {
            "status": "removed",
            "room": room_name,
            "dispatch_info": dispatch_info
        }

    print(f"⚠️ [{timestamp}] No dispatch found for room: {room_name}")
    return {
        "status": "not_found",
        "room": room_name,
        "message": f"No active dispatch found for room {room_name}"
    }


@app.get("/jobs")
async def list_jobs():
    """List all active job dispatches."""
    return {
        "active_dispatches": active_dispatches,
        "count": len(active_dispatches),
        "rooms": list(active_dispatches.keys())
    }


@app.get("/jobs/{room_name}")
async def get_job(room_name: str):
    """Get details of a specific room's agent dispatch."""
    if room_name in active_dispatches:
        return {
            "status": "active",
            "room": room_name,
            "dispatch": active_dispatches[room_name]
        }

    raise HTTPException(status_code=404, detail=f"No active dispatch found for room {room_name}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_dispatches": len(active_dispatches),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/logs/{room_name}")
async def get_chat_log(room_name: str):
    """Return chat logs for a room as a downloadable text file."""
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
        headers={"Content-Disposition": f"attachment; filename={room_name}.txt"}
    )
