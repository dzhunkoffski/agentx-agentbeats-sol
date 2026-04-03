"""A2A executor — routes incoming tasks to the Agent."""

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from agent import Agent

logger = logging.getLogger(__name__)

class Executor(AgentExecutor):
    def __init__(self):
        self.agent = Agent()

    async def execute(self, context: RequestContext, event_queue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await self.agent.run(context.message, updater)

    async def cancel(self, context: RequestContext, event_queue) -> None:
        raise NotImplementedError("Cancel not supported")
