"""A2A server entry point for τ²-Bench purple agent."""

import argparse
import logging

from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from executor import Executor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="customer-service",
        name="Customer Service Agent",
        description="Multi-turn customer service with tool use and policy compliance.",
        tags=["tau2-bench", "customer-service", "tool-calling"],
        examples=[],
    )

    agent_card = AgentCard(
        name="LangGraph τ²-Bench Agent",
        description=(
            "Customer service purple agent for τ²-Bench evaluation. "
            "Uses LangGraph for multi-turn reasoning with tool calling "
            "across airline, retail, and telecom domains."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    import uvicorn
    logger.info(f"τ²-Bench purple agent starting on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
