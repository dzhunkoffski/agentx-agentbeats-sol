"""A2A executor — routes incoming tasks to the Agent."""
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError
from a2a.utils.errors import ServerError
from agent import Agent

logger = logging.getLogger(__name__)


class Executor(AgentExecutor):
        def __init__(self):
                    self.agents: dict[str, Agent] = {}  # context_id -> agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
                context_id = context.context_id
                agent = self.agents.get(context_id)
                if not agent:
                                agent = Agent()
                                self.agents[context_id] = agent
                            await agent.run(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
                raise ServerError(error=UnsupportedOperationError())
