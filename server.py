import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import api, agents

from livekit_basic_agent import entrypoint

load_dotenv(".env")

app = FastAPI(title="LiveKit Agent Manager")

worker = agents.Worker(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="zabano_agent",
        ws_url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
)


class JobRequest(BaseModel):
    room_name: str
    agent_type: str = "tutor"
    config_schema: list


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker.run())


@app.post("/jobs")
async def create_job(request: JobRequest):
    try:
        lkapi = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        metadata = {
            "agent_type": request.agent_type,
            "source": "zabano"
        }
        js  =json.dumps(metadata)
        print(request)
        print(js)
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="zabano_agent",  # Match worker agent_name
                room=request.room_name,
                metadata=""
            )
        )
        await lkapi.aclose()

        return {
            "status": "started",
            "agent_type": request.agent_type,
            "room": request.room_name,
            "dispatch_id": dispatch.id
        }

    except Exception as e:
        print(f"❌ Dispatch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
