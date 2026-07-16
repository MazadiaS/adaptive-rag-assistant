"""FastAPI service exposing the adaptive RAG agent. Runs on port 7860 (HF Spaces)."""
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config

app = FastAPI(title="Adaptive RAG Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag_app = None


def get_rag_app():
    """Lazily build the graph so the module imports without keys (build/test)."""
    global _rag_app
    if _rag_app is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from .graph import build_rag_app
        from .ingest import get_vectorstore

        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0
        )
        retriever = get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVE_K})

        web = None
        if config.TAVILY_API_KEY:
            os.environ.setdefault("TAVILY_API_KEY", config.TAVILY_API_KEY)
            from langchain_tavily import TavilySearch
            web = TavilySearch(max_results=3)

        _rag_app = build_rag_app(llm, retriever, web)
    return _rag_app


class ChatIn(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chat_model": config.CHAT_MODEL,
        "configured": bool(config.GOOGLE_API_KEY and config.QDRANT_URL),
        "web_search": bool(config.TAVILY_API_KEY),
    }


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    from .ingest import ingest_files

    os.makedirs(config.DATA_DIR, exist_ok=True)
    paths = []
    for f in files:
        dest = os.path.join(config.DATA_DIR, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            out.write(await f.read())
        paths.append(dest)
    return ingest_files(paths)


@app.post("/chat")
def chat(body: ChatIn):
    result = get_rag_app().invoke({"question": body.question})
    return {
        "answer": result.get("generation", ""),
        "steps": result.get("steps", []),
        "sources": result.get("sources", []),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
