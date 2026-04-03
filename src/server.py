"""A2A server entry point for τ²-Bench purple agent."""

import os
import logging

from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from executor import Executor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="LangGraph τ²-Bench Agent",
        description=(
            "Customer service purple agent for τ²-Bench evaluation. "
            "Uses LangGraph for multi-turn reasoning with tool calling "
            "across airline, retail, and telecom domains."
        ),
        url=f"http://0.0.0.0:{os.getenv('AGENT_PORT', '9009')}",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="customer-service",
                name="Customer Service Agent",
                description="Multi-turn customer service with tool use and policy compliance.",
                tags=["tau2-bench", "customer-service", "tool-calling"],
            )
        ],
    )


def main():
    port = int(os.getenv("AGENT_PORT", "9009"))
    host = os.getenv("HOST", "0.0.0.0")

    card = build_agent_card()
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=None)
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    import uvicorn
    logger.info(f"τ²-Bench purple agent starting on {host}:{port}")
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    main()
