# Adaptive Multimodal Agentic RAG Assistant

Chat with your own documents — powered by a **LangGraph** agent that retrieves,
**grades its own evidence**, falls back to **web search** when the docs are weak,
and **checks its answer for hallucination** before replying. Ingests **PDFs with
images** (diagrams are captioned by Gemini Vision, so you can ask about pictures too).

Everything runs on **free tiers, no credit card**.

## The agent (LangGraph)

```
          ┌──────────┐   ┌────────────────┐
 question │ retrieve │──▶│ grade documents │
          └──────────┘   └───────┬─────────┘
                          relevant│ weak / none
                         ┌────────┴────────┐
                         ▼                 ▼
                    ┌──────────┐     ┌────────────┐
                    │ generate │◀────│ web search │ (Tavily)
                    └────┬─────┘     └────────────┘
                         ▼
                 ┌───────────────────┐
                 │ grade generation  │──▶ useful ─▶ ✅ answer + citations
                 │ (hallucination /  │──▶ not grounded ─▶ regenerate
                 │  answers question)│──▶ not useful   ─▶ web search
                 └───────────────────┘   (retry-capped, always terminates)
```

## Requirements → how they're met

| Requirement | How |
|---|---|
| RAG system | Retrieval-augmented generation over your documents |
| LangChain | Models, retriever, tools, structured-output graders |
| LangGraph | The adaptive/corrective decision graph above |
| Vector database | Qdrant Cloud (embeddings + similarity search) |
| NLP **and** Computer Vision | Text RAG **+** Gemini Vision captions images in PDFs so they're searchable |
| Deployed, free | Backend on HF Spaces, frontend on Vercel, all free tiers |
| Good frontend | Next.js chat UI that shows the agent's live steps + citations *(in `/frontend`)* |

## Verified free stack (mid-2026)

| Layer | Service | Free reality |
|---|---|---|
| LLM + Vision | Google Gemini `2.5-flash` | Free tier, no card. *If Google rotates the model, change `CHAT_MODEL`.* |
| Embeddings | Gemini `text-embedding-004` | Free, 10M tokens/min |
| Vector DB | Qdrant Cloud | Forever-free 1 GB, no card (~1M vectors) |
| Web search | Tavily | Free 1,000 searches/mo, no card (optional) |
| Backend host | Hugging Face Spaces (Docker) | Free CPU (2 vCPU/16 GB), port 7860 |
| Frontend host | Vercel | Free hobby tier |

## Deploy runbook (all free)

1. **Qdrant** — sign up at cloud.qdrant.io → create a **Free** cluster → copy its **URL** and **API key**.
2. **Gemini** — get a key at aistudio.google.com/apikey (no card).
3. **Tavily** *(optional)* — get a key at tavily.com for the web-search fallback.
4. **Backend → Hugging Face Space**
   - Create a **Docker** Space, push the `backend/` folder (its `README.md` has the Space config).
   - In Space **Settings → Secrets**, add `GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `TAVILY_API_KEY`.
   - The API comes up at `https://<user>-<space>.hf.space` — check `/health`.
5. **Frontend → Vercel**
   - Import `frontend/`, set `NEXT_PUBLIC_API_URL` to your Space URL, deploy.

## Local dev

```bash
cd backend
cp .env.example .env          # fill in your keys
uv run --with-requirements requirements.txt uvicorn app.main:app --port 7860
# open http://localhost:7860/docs
```

## Status

- ✅ **LangGraph agent** — built and unit-tested with mock models (8/8 routing tests, no keys): `backend/tests/test_graph.py`
- ✅ **Backend API** (FastAPI) — `/health`, `/ingest`, `/chat`; loads cleanly, Docker-ready
- ✅ **Multimodal ingest** — PDF text + Gemini-Vision image captions → Qdrant
- 🔜 **Frontend** (Next.js) — chat UI with live agent steps + citations
