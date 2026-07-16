"""Provider factory — returns the right LangChain LLM / embeddings for the
configured provider (OpenAI or Google Gemini). Keeps the rest of the app
provider-agnostic."""
from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage

from . import config


def get_llm(temperature: float = 0):
    if config.PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


def get_embeddings():
    if config.PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )


def caption_image(png_bytes: bytes) -> str:
    """Describe an image with the provider's vision model (CV element)."""
    b64 = base64.b64encode(png_bytes).decode()
    if config.PROVIDER == "openai":
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    else:  # gemini accepts a bare data-uri string
        image_part = {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe this image or diagram in detail for search "
                                 "indexing: what it shows, labels, numbers, and meaning."},
        image_part,
    ])
    return get_llm(temperature=0).invoke([msg]).content
