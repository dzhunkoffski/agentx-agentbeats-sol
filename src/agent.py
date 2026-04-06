"""
tau2-Bench Purple Agent - LangGraph implementation.

The green agent orchestrates multi-turn conversations. For each turn it sends
us a message containing:
  - System context (domain policies, available tools as text)
  - A user message or tool result

We must respond with JSON wrapped in <json>...</json> tags:
  {"name": "tool_name", "arguments": {...}}   -> tool call
  {"name": "respond", "arguments": {"content": "..."}}  -> respond to user
"""
import os
import json
import asyncio
import logging
import re
from typing import Annotated, TypedDict, Sequence
from a2a.utils import new_agent_text_message
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
RESPOND_ACTION_NAME = "respond"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str
    raw_input: str

AGENT_SYSTEM_PROMPT = """You are an expert customer service agent. Help customers using the tools provided and follow domain policies strictly.

RESPONSE FORMAT - You MUST respond with JSON wrapped in <json>...</json> tags:
- To call a tool:     <json>{"name": "tool_name", "arguments": {"arg1": "val1"}}</json>
- To respond to user: <json>{"name": "respond", "arguments": {"content": "your message"}}</json>

Rules:
1. ALWAYS follow domain policies.
2. Ask clarifying questions if request is ambiguous.
3. Never fabricate information.
4. Think step by step before acting.
5. Always prefer calling tools to get information before responding.
"""

def build_graph(model_name=None, temperature=0.0):
    model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    def process_message(state):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    g = StateGraph(AgentState)
    g.add_node("think", process_message)
    g.set_entry_point("think")
    g.add_edge("think", END)
    return g.compile()

def extract_json_response(text):
    match = re.search(r'<json>\s*(.*?)\s*</json>', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if "kwargs" in parsed and "arguments" not in parsed:
                parsed["arguments"] = parsed.pop("kwargs")
            return f'<json>{json.dumps(parsed)}</json>'
        except json.JSONDecodeError:
            pass
    for pattern in [r'\{[^{}]*"name"[^{}]*\}', r'\{.*?"name".*?"arguments".*?\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if "kwargs" in parsed and "arguments" not in parsed:
                    parsed["arguments"] = parsed.pop("kwargs")
                return f'<json>{json.dumps(parsed)}</json>'
            except json.JSONDecodeError:
                continue
    fallback = {"name": RESPOND_ACTION_NAME, "arguments": {"content": text}}
    return f'<json>{json.dumps(fallback)}</json>'

class Agent:
    def __init__(self):
        self.graph = build_graph()
        self.conversation = []
        logger.info("tau2-Bench purple agent initialized")

    async def run(self, context, event_queue):
        input_text = context.get_user_input()
        logger.info(f"Received: {input_text[:300]}")
        try:
            if not self.conversation:
                self.conversation.append(SystemMessage(content=AGENT_SYSTEM_PROMPT))
            self.conversation.append(HumanMessage(content=input_text))
            state = {"messages": self.conversation, "system_prompt": AGENT_SYSTEM_PROMPT, "raw_input": input_text}
            final = await asyncio.to_thread(self.graph.invoke, state)
            ai_response = ""
            for msg in reversed(final["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_response = msg.content
                    break
            formatted = extract_json_response(ai_response)
            self.conversation.append(AIMessage(content=formatted))
            logger.info(f"Responded: {formatted[:200]}")
            await event_queue.enqueue_event(
                new_agent_text_message(formatted, context_id=context.context_id)
            )
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            fallback = json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": f"Error: {e}"}})
            await event_queue.enqueue_event(
                new_agent_text_message(f"<json>{fallback}</json>", context_id=context.context_id)
          )
