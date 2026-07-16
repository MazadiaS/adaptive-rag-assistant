"""Multimodal ingestion: PDF/text -> chunks (+ image captions) -> embeddings -> Qdrant.

The CV requirement lives here: images/diagrams inside PDFs are captioned by
Gemini Vision and indexed as text, so the RAG system can answer about pictures.
"""
from __future__ import annotations

import base64
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )


def get_vectorstore():
    """Return a Qdrant-backed vector store, creating the collection if needed."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    if config.QDRANT_URL:  # Qdrant Cloud / remote server
        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    else:                  # embedded local mode — no server, no Docker, no signup
        client = QdrantClient(path=config.QDRANT_PATH)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=_embeddings(),
    )


def _caption_image(png_bytes: bytes) -> str:
    """Describe an image with Gemini Vision so its content becomes searchable."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    vision = ChatGoogleGenerativeAI(
        model=config.VISION_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0
    )
    b64 = base64.b64encode(png_bytes).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe this image or diagram in detail for search "
                                 "indexing: what it shows, labels, numbers, and meaning."},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    ])
    return vision.invoke([msg]).content


def _load_pdf(path: str, caption_images: bool = True) -> List[Document]:
    import fitz  # PyMuPDF

    docs: List[Document] = []
    name = os.path.basename(path)
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                docs.append(Document(page_content=text,
                                     metadata={"source": name, "page": page_no, "kind": "text"}))
            if caption_images:
                for img in page.get_images(full=True):
                    try:
                        pix = fitz.Pixmap(pdf, img[0])
                        if pix.n > 4:  # CMYK -> RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        caption = _caption_image(pix.tobytes("png"))
                        if caption:
                            docs.append(Document(
                                page_content=f"[Image on page {page_no}] {caption}",
                                metadata={"source": name, "page": page_no, "kind": "image"}))
                    except Exception:
                        continue  # skip unreadable images, never fail ingest
    return docs


def load_file(path: str) -> List[Document]:
    if path.lower().endswith(".pdf"):
        return _load_pdf(path)
    with open(path, "r", errors="ignore") as f:
        return [Document(page_content=f.read(),
                         metadata={"source": os.path.basename(path), "kind": "text"})]


def ingest_files(paths: List[str]) -> dict:
    raw: List[Document] = []
    for p in paths:
        raw.extend(load_file(p))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw)
    if chunks:
        get_vectorstore().add_documents(chunks)
    return {
        "files": len(paths),
        "documents": len(raw),
        "chunks": len(chunks),
        "images_captioned": sum(1 for d in raw if d.metadata.get("kind") == "image"),
    }
