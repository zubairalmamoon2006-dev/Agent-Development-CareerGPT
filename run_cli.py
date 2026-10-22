"""Chat with the agent in the terminal.
  /resume path/to/resume.pdf [optional question]   attach a resume for this session
  /quit                                            exit"""
import difflib, os, re, uuid
from careergpt.agent import build_agent


def pdf_text(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def parse_resume_arg(q):
    """/resume <path.pdf> [question] -> (path, trailing_question)."""
    rest = q[len("/resume"):].strip().strip('"')
    m = re.match(r"(?P<path>.*?\.pdf)(?:\s+(?P<q>.*))?$", rest, re.I | re.S)
    if m:
        return m.group("path").strip(), (m.group("q") or "").strip()
    return rest, ""


def resolve_path(path):
    """Exact match, else closest filename in the same folder (catches 'Resume(15).pdf')."""
    if os.path.isfile(path):
        return path, False
    folder = os.path.dirname(path) or "."
    name = os.path.basename(path)
    try:
        candidates = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    except OSError:
        return None, False
    close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.75)
    if close:
        return os.path.join(folder, close[0]), True
    return None, False


def main():
    agent = build_agent()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    session_resume = None
    print("CareerGPT ready. /resume <pdf> to attach a resume, /quit to exit.")
    while True:
        q = input("\nyou> ").strip()
        if q in ("/quit", "/exit"):
            break
        if q.startswith("/resume"):
            raw, followup = parse_resume_arg(q)
            if not raw:
                print("usage: /resume <path-to-resume.pdf>")
                continue
            path, corrected = resolve_path(raw)
            if not path:
                print(f"file not found: {raw}")
                continue
            if corrected:
                print(f"(closest match found: {os.path.basename(path)})")
            try:
                text = pdf_text(path)
            except Exception as e:
                print(f"could not read {os.path.basename(path)}: {e}")
                continue
            if not text.strip():
                print(f"{os.path.basename(path)} has no extractable text (scanned image PDF?)")
                continue
            session_resume = text
            print(f"(resume attached from {os.path.basename(path)}; "
                  "the agent loads it when you ask for personalized advice)")
            q = followup
            if not q:
                continue
        payload = {"messages": [{"role": "user", "content": q}]}
        if session_resume:
            payload["uploaded_resume_text"] = session_resume
        try:
            seen = 0
            for state in agent.stream(payload, cfg, stream_mode="values"):
                msgs = state["messages"]
                for m in msgs[seen:]:
                    for tc in getattr(m, "tool_calls", None) or []:
                        print(f"  -> {tc['name']}({tc['args']})")
                seen = len(msgs)
            print("\ncareergpt>", getattr(msgs[-1], "text", msgs[-1].content))
        except Exception as e:
            print(f"(error: {e})")


if __name__ == "__main__":
    main()
