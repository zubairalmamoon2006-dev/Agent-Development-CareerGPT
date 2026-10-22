import json, os, time
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt, wrap_tool_call
from langchain_core.messages import ToolMessage

from .config import TOOL_LOG_PATH
from .prompts import BASE_PROMPT, RESUME_AWARE_PROMPT


class CareerState(AgentState):
    uploaded_resume_text: NotRequired[str]   # set by the app when a student uploads a resume
    resume_loaded: NotRequired[bool]         # set by load_user_resume


class CareerStateMiddleware(AgentMiddleware[CareerState]):
    """Registers the extra state fields on the agent."""
    state_schema = CareerState


@dynamic_prompt
def resume_aware_prompt(request: ModelRequest) -> str:
    """Dynamic prompt switching: resume-aware prompt only after load_user_resume succeeded."""
    state = request.state
    if state.get("resume_loaded"):
        return RESUME_AWARE_PROMPT.format(resume=state.get("uploaded_resume_text", ""))
    return BASE_PROMPT


@wrap_tool_call
def log_tool_calls(request, handler):
    """Tool-call monitoring: JSONL log of name, args, success/failure, latency."""
    call = request.tool_call
    t0 = time.time()
    status, error, result = "success", None, None
    try:
        result = handler(request)
    except Exception as e:  # keep the agent alive; let it see the error
        status, error = "failure", str(e)
        result = ToolMessage(f"Tool error: {e}", tool_call_id=call["id"], status="error")
    os.makedirs(os.path.dirname(TOOL_LOG_PATH) or ".", exist_ok=True)
    with open(TOOL_LOG_PATH, "a") as f:
        f.write(json.dumps({"ts": time.time(), "tool": call["name"], "args": call["args"],
                            "status": status, "error": error,
                            "latency_s": round(time.time() - t0, 3)}) + "\n")
    return result
