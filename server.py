import json
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from livekit import api, agents
from livekit_basic_agent import entrypoint

# Load environment variables
load_dotenv(".env")

app = FastAPI(title="LiveKit Agent Manager")

# Track active dispatches per room with details
active_dispatches = {}

# Locks for preventing race conditions per room
dispatch_locks = {}

# Initialize the worker
worker = agents.Worker(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="zabano_agent",
        ws_url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
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
    """Start the LiveKit worker on app startup."""
    print("🚀 Starting LiveKit Agent Worker...")
    asyncio.create_task(worker.run())
    print("✅ Worker started successfully")


@app.post("/jobs")
async def create_job(request: JobRequest):
    """
    Spawn a new agent in a specific room.

    Args:
        request: JobRequest with room_name, agent_type, and optional config

    Returns:
        dict with status, agent_type, room, and dispatch_id
    """
    timestamp = datetime.now().isoformat()
    print(f"🔔 [{timestamp}] Received job request:")
    print(f"   Room: {request.room_name}")
    print(f"   Agent Type: {request.agent_type}")
    print(f"   Config: {request.config}")

    # Get lock for this room to prevent race conditions
    lock = await get_room_lock(request.room_name)

    async with lock:
        # Check if agent already exists for this room

        try:
            # Create LiveKit API client
            lkapi = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )

            print(f"📤 [{timestamp}] Creating dispatch for room {request.room_name}...")

            # Prepare metadata
            metadata_dict = {
                "agent_type": request.agent_type,
                "source": "zabano"
            }

            # Add config if provided
            if request.config:
                metadata_dict["config"] = request.config

            # Create the agent dispatch
            dispatch = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="zabano_agent",
                    room=request.room_name,
                    metadata=json.dumps(metadata_dict)
                )
            )

            await lkapi.aclose()

            # Track this dispatch
            active_dispatches[request.room_name] = {
                'agent_type': request.agent_type,
                'dispatch_id': dispatch.id,
                'timestamp': timestamp,
                'config': request.config
            }

            print(f"✅ [{timestamp}] Dispatch created successfully!")
            print(f"   Dispatch ID: {dispatch.id}")
            print(f"   Room: {request.room_name}")
            print(f"   Agent Type: {request.agent_type}")

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
    """
    Remove tracking for a room.
    Call this when a room closes or agent should be removed.

    Args:
        room_name: Name of the room to remove tracking for

    Returns:
        dict with status and room name
    """
    timestamp = datetime.now().isoformat()

    if room_name in active_dispatches:
        dispatch_info = active_dispatches[room_name]
        del active_dispatches[room_name]

        print(f"🗑️ [{timestamp}] Removed dispatch tracking for room: {room_name}")
        print(f"   Agent Type: {dispatch_info['agent_type']}")
        print(f"   Dispatch ID: {dispatch_info['dispatch_id']}")

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
    """
    List all active job dispatches.

    Returns:
        dict with active_dispatches and count
    """
    return {
        "active_dispatches": active_dispatches,
        "count": len(active_dispatches),
        "rooms": list(active_dispatches.keys())
    }


@app.get("/jobs/{room_name}")
async def get_job(room_name: str):
    """
    Get details of a specific room's agent dispatch.

    Args:
        room_name: Name of the room

    Returns:
        dict with dispatch details or 404
    """
    if room_name in active_dispatches:
        return {
            "status": "active",
            "room": room_name,
            "dispatch": active_dispatches[room_name]
        }

    raise HTTPException(
        status_code=404,
        detail=f"No active dispatch found for room {room_name}"
    )


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
    log_dir = "chat_logs"
    file_path = os.path.join(log_dir, f"{room_name}.txt")

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"No log file found for room '{room_name}'"
        )

    return FileResponse(
        path=file_path,
        filename=f"{room_name}.txt",
        media_type="text/plain"
    )
