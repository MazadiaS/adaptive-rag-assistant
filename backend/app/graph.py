"""Adaptive / Corrective RAG agent built with LangGraph.

Flow:
    retrieve -> grade_documents -> (web_search?) -> generate
             -> grade_generation -> (END | regenerate | web_search)

Everything the graph needs (llm, retriever, web_search_tool) is injected, so the
graph can be unit-tested with fakes and NO API keys — see tests/test_graph.py.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

MAX_RETRIES = 2


class GraphState(TypedDict, total=False):
    question: str
    documents: List[Document]
    generation: str
    web_search: bool
    retries: int
    steps: List[str]          # execution trace, surfaced to the frontend
    sources: List[dict]       # citations for the frontend


# ---- structured grader schemas ----
class YesNo(BaseModel):
    binary_score: str = Field(description="Answer 'yes' or 'no' only.")


def _yes(x) -> bool:
    return "yes" in str(getattr(x, "binary_score", x)).lower()


def _format(docs: List[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def build_rag_app(llm, retriever, web_search_tool=None):
    """Return a compiled LangGraph app. `llm` must support with_structured_output()."""

    def _grader(system: str):
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{input}")]
        )
        return prompt | llm.with_structured_output(YesNo)

    doc_grader = _grader(
        "You assess whether a retrieved document is relevant to the user's question. "
        "Answer 'yes' if it contains information that could help answer it, else 'no'."
    )
    hallucination_grader = _grader(
        "You assess whether an answer is grounded in / supported by the given documents. "
        "Answer 'yes' if every claim is supported, else 'no'."
    )
    answer_grader = _grader(
        "You assess whether an answer actually resolves the user's question. "
        "Answer 'yes' or 'no'."
    )

    gen_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise assistant. Answer the question using ONLY the "
                    "context below. If the context is not enough, say you don't know. "
                    "Cite the sources you use as [n].",
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # ---------------- nodes ----------------
    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])
        return {
            "documents": docs,
            "retries": state.get("retries", 0),
            "steps": state.get("steps", []) + ["retrieve"],
        }

    def grade_documents(state: GraphState) -> dict:
        kept: List[Document] = []
        for d in state["documents"]:
            verdict = doc_grader.invoke(
                {"input": f"Question: {state['question']}\nDocument: {d.page_content}"}
            )
            if _yes(verdict):
                kept.append(d)
        return {
            "documents": kept,
            "web_search": len(kept) == 0,
            "steps": state.get("steps", []) + ["grade_documents"],
        }

    def web_search(state: GraphState) -> dict:
        docs = list(state["documents"])
        if web_search_tool is not None:
            try:
                res = web_search_tool.invoke({"query": state["question"]})
                if isinstance(res, str):
                    text = res
                else:  # Tavily returns a list of {"content","url",...}
                    text = "\n\n".join(r.get("content", "") for r in res)
                docs.append(Document(page_content=text, metadata={"source": "web"}))
            except Exception as e:  # never crash the graph on a flaky search
                docs.append(
                    Document(
                        page_content=f"(web search unavailable: {e})",
                        metadata={"source": "web"},
                    )
                )
        return {"documents": docs, "steps": state.get("steps", []) + ["web_search"]}

    def generate(state: GraphState) -> dict:
        answer = gen_chain.invoke(
            {"question": state["question"], "context": _format(state["documents"])}
        )
        sources = [
            {"n": i + 1, "source": d.metadata.get("source", "document"),
             "preview": d.page_content[:160]}
            for i, d in enumerate(state["documents"])
        ]
        return {
            "generation": answer,
            "sources": sources,
            "retries": state.get("retries", 0) + 1,
            "steps": state.get("steps", []) + ["generate"],
        }

    # ---------------- conditional edges ----------------
    def route_after_grade(state: GraphState) -> str:
        return "web_search" if state.get("web_search") else "generate"

    def route_after_generate(state: GraphState) -> str:
        # stop looping after MAX_RETRIES so we always terminate
        if state.get("retries", 0) >= MAX_RETRIES:
            return "useful"
        grounded = hallucination_grader.invoke(
            {"input": f"Documents:\n{_format(state['documents'])}\n\nAnswer: {state['generation']}"}
        )
        if not _yes(grounded):
            return "not_grounded"      # hallucinated -> regenerate
        useful = answer_grader.invoke(
            {"input": f"Question: {state['question']}\nAnswer: {state['generation']}"}
        )
        return "useful" if _yes(useful) else "not_useful"  # not_useful -> widen with web

    def route_after_generate_docs_only(state: GraphState) -> str:
        # document-only: never widen with the web; end instead
        return "not_grounded" if route_after_generate(state) == "not_grounded" else "useful"

    # ---------------- wire the graph ----------------
    web_enabled = web_search_tool is not None

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")

    if web_enabled:
        # agentic mode: weak evidence / unhelpful answer -> fall back to the web
        g.add_node("web_search", web_search)
        g.add_conditional_edges(
            "grade_documents", route_after_grade,
            {"web_search": "web_search", "generate": "generate"},
        )
        g.add_edge("web_search", "generate")
        g.add_conditional_edges(
            "generate", route_after_generate,
            {"useful": END, "not_grounded": "generate", "not_useful": "web_search"},
        )
    else:
        # document-only mode: answer from the documents or say "I don't know" — never the web
        g.add_edge("grade_documents", "generate")
        g.add_conditional_edges(
            "generate", route_after_generate_docs_only,
            {"useful": END, "not_grounded": "generate"},
        )
    return g.compile()
