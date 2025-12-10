import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "data" / "docs"
CODE_DIR = BASE_DIR / "data" / "code_snippets"
LOGS_DIR = BASE_DIR / "data" / "logs"

INDEX_DIR = BASE_DIR / "data" / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEXT_MODEL = "gpt-5.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 5
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

MIN_SIMILARITY_THRESHOLD = 0.35

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Set it before running the app.")
