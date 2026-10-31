# CareerGPT evaluation harness

Measures agent **behavior** (which tools fire, which facts appear), not code correctness.
`pytest` covers the code; this covers the answers.

## Run
```powershell
python eval/run_eval.py                 # full run against Gemini (~42 calls, needs GOOGLE_API_KEY)
python eval/run_eval.py --filter multi_part
python eval/run_eval.py --limit 5
python eval/run_eval.py --dry-run       # validate questions.jsonl, no LLM
python eval/run_eval.py --fake          # offline plumbing check with a scripted model, no key
```
Results: printed table + `eval/results/<timestamp>-<model>.json` (gitignored).
First real run = **the baseline**. Re-run after every prompt/tool-docstring change and
record the delta here before keeping the change.

## Question file — `questions.jsonl`
One JSON object per line:
```json
{"id": "q01", "category": "company_role", "query": "Which companies come for analytics internships?",
 "expected_tools": ["search_placement_blogs"], "must_mention": ["ExampleCorp"]}
```
| Field | Meaning |
| --- | --- |
| `expected_tools` | tools that MUST be called (empty list = none required) |
| `forbidden_tools` | tools that must NOT be called (used by negative questions) |
| `must_mention` | substrings that must appear in the answer (case-insensitive) |
| `must_not_mention` | substrings that must NOT appear (catches fabrication) |

A question **passes** only if all tool rules AND all term rules hold.
Categories: `company_role`, `timeline_process`, `preparation`, `project_por`, `policy`,
`multi_part` (≥2 tools), `negative` (off-topic / unanswerable — must not fabricate),
`resume_gate` (asks for resume, none attached — must ask for an upload).

**When the real data lands:** the `must_mention` terms were written against the dummy
passages in `seed_dummy_data.py`. Sweep the file and replace them with real facts from
the indexed data (company names, dates) — the queries themselves stay.

## Decision log
| Date | Change | Overall | Notes |
| --- | --- | --- | --- |
| 2026-10-31 | baseline on `gemini-3.5-flash-lite` | 40/42 = 95% | `project_por_06` was a keyword artifact (`ML` vs "Machine learning" — keyword fixed, would be 41/42); real failure: `resume_gate_02` (answered generically + offered upload instead of calling `load_user_resume` first). `gemini-2.5-flash` unusable for eval: free tier = 20 req/day. |

## For the data/indexing team — `gold_passages.jsonl`
Line format:
```json
{"query": "when does shortlisting for internships begin", "collection": "blogs", "gold_ids": ["dummy-blogs-1"]}
```
These `(query → gold passage ids)` pairs are what lets us measure **retrieval quality**
(recall@k, MRR) and jointly pick the embedding model and `TOP_K`.
Current ids reference `seed_dummy_data.py` placeholders (`dummy-<collection>-<n>`).
After real indexing, regenerate this file with **real passage ids** — same format,
roughly this coverage (a few dozen queries spread over the four collections).
