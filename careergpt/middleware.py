"""Middleware: state registration, dynamic resume prompt, tool-call logging,
and grounding (citation) enforcement around the model call."""
import json, os, re, time
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import (AgentMiddleware, ModelRequest, ModelResponse,
                                         dynamic_prompt, wrap_model_call, wrap_tool_call)
from langchain_core.messages import HumanMessage, ToolMessage

from .config import MAX_RESUME_CHARS, TOOL_LOG_PATH
from .prompts import BASE_PROMPT, RESUME_AWARE_PROMPT


class CareerState(AgentState):
    uploaded_resume_text: NotRequired[str]   # set by the app when a student uploads a resume
    resume_loaded: NotRequired[bool]         # set by load_user_resume


class CareerStateMiddleware(AgentMiddleware[CareerState]):
    """Registers the extra state fields on the agent."""
    state_schema = CareerState


# --------------------------------------------------------------------------- #
# helpers (pure, unit-tested without any LLM)
# --------------------------------------------------------------------------- #
def truncate_resume(text: str, limit: int = MAX_RESUME_CHARS) -> str:
    text = text or ""
    if len(text) > limit:
        return text[:limit] + "\n[resume truncated]"
    return text


def pick_prompt(state: dict) -> str:
    """Resume-aware prompt only after load_user_resume succeeded."""
    if state.get("resume_loaded"):
        return RESUME_AWARE_PROMPT.format(resume=truncate_resume(state.get("uploaded_resume_text", "")))
    return BASE_PROMPT


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)


def extract_sources(messages) -> set[str]:
    """Source labels (e.g. placement_blog_2025) present in tool results."""
    sources = set()
    for m in messages or []:
        if getattr(m, "type", "") == "tool":
            sources.update(re.findall(r"source=([\w.\-]+)", _message_text(m)))
    return sources


def answer_cites(answer: str, sources: set[str]) -> bool:
    a = (answer or "").lower()
    return any(s.lower() in a for s in sources)


def _append_log(payload: dict) -> None:
    os.makedirs(os.path.dirname(TOOL_LOG_PATH) or ".", exist_ok=True)
    with open(TOOL_LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")


def write_tool_log(call: dict, status: str, error, latency_s: float) -> None:
    """JSONL log line. Args are redacted when long: resume/answer text never lands here."""
    redacted = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 300 else v)
                for k, v in (call.get("args") or {}).items()}
    _append_log({"ts": time.time(), "event": "tool_call", "tool": call.get("name"),
                 "args": redacted, "status": status, "error": error,
                 "latency_s": round(latency_s, 3)})


# One nudge per user turn, tracked by the id of the last human message.
# The nudged message itself is NOT persisted to graph state (it only exists in the
# overridden request), so content-matching alone cannot stop a model that keeps
# replying with tool calls — this cap is the durable guard against an infinite
# nudge -> tool -> answer -> nudge loop.
_nudged_turns: dict[str, int] = {}


def _turn_key(messages) -> str | None:
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human":
            return getattr(m, "id", None) or _message_text(m)[:200]
    return None


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #
@dynamic_prompt
def resume_aware_prompt(request: ModelRequest) -> str:
    """Dynamic prompt switching: resume-aware prompt only after load_user_resume succeeded."""
    return pick_prompt(request.state)


@wrap_tool_call
def log_tool_calls(request, handler):
    """Tool-call monitoring: name, args (redacted), success/failure, latency."""
    call = request.tool_call
    t0 = time.time()
    status, error, result = "success", None, None
    try:
        result = handler(request)
    except Exception as e:  # keep the agent alive; let it see the error
        status, error = "failure", str(e)
        result = ToolMessage(f"Tool error: {e}", tool_call_id=call["id"], status="error")
    write_tool_log(call, status, error, time.time() - t0)
    return result


@wrap_model_call
def grounding_check(request, handler):
    """Citation enforcement: if retrieval happened, the final answer must cite a source.
    One corrective regeneration; never loops. Logs 'grounding_fix' events."""
    response = handler(request)
    messages = response.result if isinstance(response, ModelResponse) else [response]
    final = messages[-1] if messages else None
    if final is None or getattr(final, "tool_calls", None):
        return response  # intermediate step (tool calls pending) or empty response

    sources = extract_sources(request.state.get("messages", []))
    answer = _message_text(final)
    if not sources or answer_cites(answer, sources):
        return response

    nudge_text = ("Your last answer did not cite any source label from the retrieved passages.")
    turn = _turn_key(request.messages)
    if turn and _nudged_turns.get(turn, 0) >= 1:
        return response  # already nudged once this turn — never loop
    if any(nudge_text in _message_text(m) for m in (request.messages or [])):
        return response  # nudge visible in history (defensive, same guarantee)

    if turn:
        _nudged_turns[turn] = _nudged_turns.get(turn, 0) + 1
        if len(_nudged_turns) > 500:  # bound memory: keep the most recent turns
            for old in list(_nudged_turns)[:250]:
                _nudged_turns.pop(old, None)

    nudge = HumanMessage(content=(
        nudge_text + " Rewrite it, citing the source (e.g. source=placement_blog_2025) "
        "for each factual claim, and do not add any new facts."))
    _append_log({"ts": time.time(), "event": "grounding_fix", "status": "regenerated",
                 "sources": sorted(sources)})
    return handler(request.override(messages=[*request.messages, nudge]))
