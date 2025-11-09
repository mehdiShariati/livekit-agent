import json
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import api  # server SDK for dispatch
from livekit import agents  # worker side (unchanged)

from livekit_basic_agent import entrypoint  # import your dynamic entrypoint

app = FastAPI(title="LiveKit Agent Manager")

# Initialize the worker (this runs the agent code)
worker = agents.Worker(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        # optionally set agent_name if you want explicit agent names
        # agent_name="onboarding_agent"
    )
)

class JobRequest(BaseModel):
    room_name: str
    agent_type: str = "tutor"  # default

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker.run())

@app.post("/jobs")
async def create_job(request: JobRequest):
    """Spawn a new agent in a specific room."""
    try:
        # Create a dispatch via LiveKit server
        lkapi = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )

        # metadata can be JSON string
        metadata = {"agent_type": request.agent_type}

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=request.agent_type,  # must match your AgentTemplate key/agent_name
                room=request.room_name,
                metadata="salam chetori?",
            )
        )
        await lkapi.aclose()

        return {
            "status": "started",
            "agent_type": request.agent_type,
            "room": request.room_name,
        }

    except Exception as e:
        # optionally log e
        raise HTTPException(status_code=500, detail=str(e))
