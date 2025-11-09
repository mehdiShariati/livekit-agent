"""
server.py — LiveKit Agent Job Manager
-------------------------------------
This file adds an HTTP endpoint to spawn dynamic agents
with metadata (room_name, agent_type, etc.)
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import agents
from livekit_basic_agent import entrypoint  # import your dynamic entrypoint

app = FastAPI(title="LiveKit Agent Manager")

# ✅ Correct Worker initialization
worker = agents.Worker(
    agents.WorkerOptions(entrypoint_fnc=entrypoint)
)

class JobRequest(BaseModel):
    room_name: str
    agent_type: str = "tutor"  # default

@app.on_event("startup")
async def startup_event():
    # Start the LiveKit worker in background
    asyncio.create_task(worker.run())

@app.post("/jobs")
async def create_job(request: JobRequest):
    """Spawn a new LiveKit Agent for a specific room & agent_type"""
    try:
        metadata = {"agent_type": request.agent_type}

        await worker.create_job(
            metadata=metadata,
            room=request.room_name,
        )

        return {
            "status": "started",
            "agent_type": request.agent_type,
            "room": request.room_name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
