import asyncio

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer

from livekit_basic_agent import entrypoint

load_dotenv(".env")

server = AgentServer()
server.rtc_session(entrypoint, agent_name="zabano_agent")


if __name__ == "__main__":
    agents.cli.run_app(server)
