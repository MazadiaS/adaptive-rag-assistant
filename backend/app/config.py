"""All configuration comes from environment variables (set as secrets in
Hugging Face Spaces / a local .env). Nothing secret is committed.

Provider is auto-detected: if OPENAI_API_KEY is set -> OpenAI, else Google Gemini.
Override with LLM_PROVIDER=openai|google.
"""
import os

# Load a local .env if present (harmless in production where real env vars win).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _detect_provider() -> str:
    p = os.getenv("LLM_PROVIDER")
    if p:
        return p.lower()
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "google"


PROVIDER = _detect_provider()

# --- Google Gemini (free tier) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")       # cheap + vision
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Embedding dimension depends on the provider's embedding model.
_DEFAULT_DIM = 1536 if PROVIDER == "openai" else 768
EMBED_DIM = int(os.getenv("EMBED_DIM", str(_DEFAULT_DIM)))

# --- Vector DB (Qdrant) ---
QDRANT_URL = os.getenv("QDRANT_URL")            # empty -> embedded local (no server)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", "/tmp/qdrant_local")
# collection name carries the dim so switching providers never causes a size mismatch
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", f"documents_{EMBED_DIM}")

# --- web search (optional) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- retrieval / chunking ---
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

DATA_DIR = os.getenv("DATA_DIR", "/tmp/uploads")
