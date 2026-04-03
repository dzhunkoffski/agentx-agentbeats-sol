"""Messenger utility for A2A agent-to-agent communication."""
import logging
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)

class Messenger:
    async def talk_to_agent(self, message: str, agent_url: str) -> str:
        try:
            client = await A2AClient.get_client_from_agent_card_url(
                f"{agent_url}/.well-known/agent.json"
            )
            request = SendMessageRequest(
                params=MessageSendParams(message=new_agent_text_message(message))
            )
            response = await client.send_message(request)
            if hasattr(response, "result") and hasattr(response.result, "artifacts"):
                for artifact in (response.result.artifacts or []):
                    for part in (artifact.parts or []):
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            return part.root.text
            return str(response)
        except Exception as e:
            logger.error(f"Error talking to agent at {agent_url}: {e}")
            return f"Communication error: {e}"
