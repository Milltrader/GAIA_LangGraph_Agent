
import logging
import os
import sys

# ── Settings ──────────────────────────────────────────────────
QUESTIONS_URL   = os.getenv("QUESTIONS_URL",
    "https://agents-course-unit4-scoring.hf.space/questions")
SUBMIT_URL      = os.getenv("SUBMIT_URL",
    "https://agents-course-unit4-scoring.hf.space/submit")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")       # required for OpenAI
EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 4))
DEBUG           = os.getenv("DEBUG", "false").lower() == "true"

if OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

_LOGGING_READY = False          # module-level guard

def get_logger(name: str | None = None) -> logging.Logger:
    """
    First call sets up root logger; subsequent calls just return a child.
    """
    global _LOGGING_READY
    if not _LOGGING_READY:
        debug_flag = os.getenv("DEBUG", "false").lower() == "true"
        level = logging.DEBUG if debug_flag else logging.INFO
        logging.basicConfig(
            stream=sys.stdout,                 # switch to stderr if preferred
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGING_READY = True
    return logging.getLogger(name or __name__)
