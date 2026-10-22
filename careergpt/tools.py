"""Five tools. The docstrings ARE the routing logic: the agent picks tools from them,
so edit them carefully (there is no classifier)."""
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .store import search


@tool
def search_placement_blogs(query: str) -> str:
    """Search placement and internship blogs and portal records (current and previous years).
    Use for: which companies visit campus, roles offered, job descriptions, domains a company
    hires for, recruitment timelines (when shortlisting starts, order of tests/interviews),
    stipend or compensation. Query should be a short natural-language search."""
    return search("blogs", query)


@tool
def search_career_booklets(query: str) -> str:
    """Search Career Cell booklets (Internship Guide 101, Placement Blue Book, University
    Internship Booklet, GRAD Booklet, Core Internship Preparation Booklet).
    Use for: preparation guidance by domain, expected skills, resume tips, interview advice,
    research internships, higher-studies guidance, general process timelines."""
    return search("booklets", query)


@tool
def search_resume_patterns(query: str) -> str:
    """Search the anonymized resume repository for patterns by domain.
    Use for: what projects, skills and positions of responsibility (PORs) are common for a
    domain (data, consulting, finance, software, core), and project/POR suggestions."""
    return search("resumes", query)


@tool
def search_institute_policies(query: str) -> str:
    """Search institute rules, reports and policy pages on internships and placements.
    Use for: placement registration requirements, resume verification guidelines,
    eligibility, official institute processes and rules."""
    return search("policies", query)


@tool
def load_user_resume(runtime: ToolRuntime) -> Command:
    """Load the student's uploaded resume for this session so advice can be personalized.
    Call ONLY when the student asks for feedback or suggestions based on their own resume.
    Takes no arguments. Reports whether a resume was found."""
    text = (runtime.state.get("uploaded_resume_text") or "").strip()
    if not text:
        return Command(update={"messages": [ToolMessage(
            "No resume uploaded in this session. Ask the student to upload one.",
            tool_call_id=runtime.tool_call_id)]})
    return Command(update={
        "resume_loaded": True,  # flips the prompt to resume-aware mode (see middleware)
        "messages": [ToolMessage("Resume loaded. You may now reference it.",
                                 tool_call_id=runtime.tool_call_id)],
    })


ALL_TOOLS = [search_placement_blogs, search_career_booklets, search_resume_patterns,
             search_institute_policies, load_user_resume]
