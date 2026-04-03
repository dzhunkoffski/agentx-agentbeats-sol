"""
τ²-Bench Purple Agent — LangGraph implementation.

The green agent orchestrates multi-turn conversations. For each turn it sends
us a message containing:
  - System context (domain policies, available tools as text)
  - A user message or tool result

We must respond with JSON wrapped in <json>...</json> tags:
  {"name": "tool_name", "kwargs": {...}}          → tool call
  {"name": "respond", "kwargs": {"content": "…"}} → respond to user

This agent uses LangGraph to:
1. Parse the green agent's message format
2. Decide whether to call a tool or respond
3. Format the output correctly
"""

import os
import json
import asyncio
import logging
import re
from typing import Annotated, TypedDict, Sequence, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from messenger import Messenger

logger = logging.getLogger(__name__)

# The action name used when responding to the user (not calling a tool)
RESPOND_ACTION_NAME = "respond"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str
    raw_input: str


# ---------------------------------------------------------------------------
# System prompt — injected as first message, then the green agent's
# tool list and user messages are appended each turn.
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are an expert customer service agent. You help customers with their \
requests by using the tools provided and following domain policies strictly.

CRITICAL RULES:
1. ALWAYS follow domain policies. Never violate a policy to satisfy a request.
2. If a request is ambiguous or missing information, ask a clarifying question.
3. If you lack the capability or data to fulfill a request, say so. Never \
   fabricate information or pretend you can do something you cannot.
4. When the user needs to take an action (e.g., turn on mobile data), clearly \
   instruct them what to do.
5. Be concise but complete. Address every part of the customer's request.
6. Think step by step before deciding your action.

RESPONSE FORMAT:
You MUST respond with JSON wrapped in <json>...</json> tags.
- To call a tool:  <json>{"name": "tool_name", "kwargs": {"arg1": "val1"}}</json>
- To respond to user: <json>{"name": "respond", "kwargs": {"content": "your message"}}</json>

Only call ONE tool at a time. After a tool result, decide your next action.
Always prefer getting information via tools before responding to the user.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def build_graph(model_name: str = None, temperature: float = 0.0):
    """Build the LangGraph for τ²-Bench interactions."""

    model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    def process_message(state: AgentState) -> dict:
        """Single LLM call — the green agent drives the multi-turn loop."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(AgentState)
    g.add_node("think", process_message)
    g.set_entry_point("think")
    g.add_edge("think", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json_response(text: str) -> str:
    """
    Ensure the response contains properly formatted <json>...</json>.
    If the LLM already included it, extract and re-wrap cleanly.
    If not, try to find JSON in the response and wrap it.
    """
    # Already has <json> tags
    match = re.search(r'<json>\s*(.*?)\s*</json>', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return f'<json>{json.dumps(parsed)}</json>'
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return f'<json>{json.dumps(parsed)}</json>'
        except json.JSONDecodeError:
            pass

    # Try to find JSON with nested kwargs
    json_match = re.search(r'\{.*?"name".*?"kwargs".*?\}(?:\s*\})?', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return f'<json>{json.dumps(parsed)}</json>'
        except json.JSONDecodeError:
            pass

    # Fallback: wrap the entire text as a respond action
    logger.warning(f"Could not extract JSON, wrapping as respond: {text[:200]}")
    fallback = {"name": RESPOND_ACTION_NAME, "kwargs": {"content": text}}
    return f'<json>{json.dumps(fallback)}</json>'


# ---------------------------------------------------------------------------
# A2A Agent class
# ---------------------------------------------------------------------------

class Agent:
    """Purple agent for τ²-Bench on AgentBeats."""

    def __init__(self):
        self.messenger = Messenger()
        self.graph = build_graph()
        self.conversation: list[BaseMessage] = []
        logger.info("τ²-Bench purple agent (LangGraph) initialized")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Called by A2A executor for each message from the green agent.

        The green agent drives a multi-turn loop:
          Turn 1: system instructions + tool list + first user message
          Turn N: tool results or next user message
          ...until task is resolved.
        """
        input_text = get_message_text(message)
        logger.info(f"Received: {input_text[:300]}...")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing..."),
        )

        try:
            # Build message list for this turn
            # On first turn, prepend our system prompt
            if not self.conversation:
                self.conversation.append(
                    SystemMessage(content=AGENT_SYSTEM_PROMPT)
                )

            # Add the green agent's message as a human message
            self.conversation.append(HumanMessage(content=input_text))

            # Run through LangGraph
            state = {"messages": self.conversation, "system_prompt": AGENT_SYSTEM_PROMPT, "raw_input": input_text}
            final = await asyncio.to_thread(self.graph.invoke, state)

            # Extract the AI response
            ai_response = ""
            for msg in reversed(final["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_response = msg.content
                    break

            # Ensure proper JSON format
            formatted = extract_json_response(ai_response)

            # Track conversation for multi-turn
            self.conversation.append(AIMessage(content=formatted))

            # Return to green agent
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=formatted))],
                name="response",
            )
            logger.info(f"Responded: {formatted[:200]}")

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            # Even on error, respond in the expected format
            fallback = json.dumps({
                "name": RESPOND_ACTION_NAME,
                "kwargs": {"content": f"I apologize, I encountered an error: {e}"}
            })
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"<json>{fallback}</json>"))],
                name="response",
            )
