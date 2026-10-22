from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from .config import GEMINI_MODEL
from .middleware import CareerStateMiddleware, log_tool_calls, resume_aware_prompt
from .prompts import BASE_PROMPT
from .tools import ALL_TOOLS


def build_agent(model=None):
    """create_agent compiles the LangGraph model<->tool loop (ReAct).
    Pass any chat model that supports tool calling to swap Gemini out."""
    model = model or ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=BASE_PROMPT,  # overridden per step by resume_aware_prompt
        middleware=[CareerStateMiddleware(), resume_aware_prompt, log_tool_calls],
        checkpointer=InMemorySaver(),  # per-session memory via thread_id
    )
