"""All configuration comes from environment variables (set as secrets in
Hugging Face Spaces / a local .env). Nothing secret is committed."""
import os

# --- models (free tier). Update the model name if Google rotates it. ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")        # free Flash tier
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")    # same model, multimodal
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))                  # text-embedding-004 = 768

# --- keys / endpoints ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# writable dir (Hugging Face Spaces only allow /tmp)
DATA_DIR = os.getenv("DATA_DIR", "/tmp/uploads")
