"""Set test env BEFORE any careergpt module is imported.

test_eval.py (alphabetically first) imports careergpt.agent -> config, so the
temp-dir env vars set at the top of test_offline.py would otherwise arrive too
late and tests would read/write the REAL chroma_db.
"""
import os
import tempfile

os.environ.setdefault("CHROMA_DIR", tempfile.mkdtemp(prefix="careergpt_test_chroma_"))
os.environ.setdefault("TOOL_LOG_PATH",
                      os.path.join(tempfile.mkdtemp(prefix="careergpt_test_logs_"),
                                   "log.jsonl"))
