---
title: Adaptive RAG Assistant API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Adaptive RAG Assistant — Backend

FastAPI + LangGraph adaptive/corrective RAG. Deployed as a Docker Space.

Set these as **Space secrets** (Settings → Variables and secrets):
`GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `TAVILY_API_KEY` (optional).

Endpoints: `GET /health`, `POST /ingest` (multipart files), `POST /chat` (`{"question": "..."}`).
