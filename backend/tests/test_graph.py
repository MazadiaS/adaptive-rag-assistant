"""Runs the LangGraph agent with FAKE models (no API keys) to prove routing.

Run:  uv run --with langgraph --with langchain-core --with pydantic python backend/tests/test_graph.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.graph import build_rag_app


class FakeLLM(Runnable):
    """Minimal stand-in: pipeable, and supports with_structured_output()."""
    def __init__(self, gen_text="The answer is 42 [1].", doc="yes", hallu="yes", ans="yes"):
        self.gen_text, self.doc, self.hallu, self.ans = gen_text, doc, hallu, ans

    def invoke(self, input, config=None, **kw):
        return AIMessage(content=self.gen_text)

    def with_structured_output(self, schema, **kw):
        o = self
        def _run(pv):
            t = (pv.to_string() if hasattr(pv, "to_string") else str(pv)).lower()
            score = o.ans if "resolves" in t else o.hallu if ("grounded" in t or "supported" in t) else o.doc
            return schema(binary_score=score)
        return RunnableLambda(_run)


class FakeRetriever:
    def __init__(self, docs): self.docs = docs
    def invoke(self, q): return self.docs


class FakeWeb:
    def invoke(self, x): return [{"content": "web result about " + x["query"], "url": "http://x"}]


DOCS = [Document(page_content="doc a", metadata={"source": "a.pdf"}),
        Document(page_content="doc b", metadata={"source": "b.pdf"})]

results = []
def check(name, cond):
    results.append(cond)
    print(("PASS" if cond else "FAIL"), "-", name)


# 1) happy path: relevant docs -> generate -> useful -> END
app = build_rag_app(FakeLLM(doc="yes", hallu="yes", ans="yes"), FakeRetriever(DOCS), FakeWeb())
out = app.invoke({"question": "what is the answer?"})
check("happy path steps = retrieve,grade,generate", out["steps"] == ["retrieve", "grade_documents", "generate"])
check("happy path produced an answer", bool(out.get("generation")))
check("happy path did NOT use web search", "web_search" not in out["steps"])
check("happy path returns citations", len(out.get("sources", [])) == 2)

# 2) weak retrieval: all docs graded irrelevant -> web_search fallback
app2 = build_rag_app(FakeLLM(doc="no", hallu="yes", ans="yes"), FakeRetriever(DOCS), FakeWeb())
out2 = app2.invoke({"question": "obscure question"})
check("weak retrieval falls back to web_search", "web_search" in out2["steps"])
check("weak retrieval still answers", bool(out2.get("generation")))

# 3) self-correction: hallucinated answer -> regenerate, then terminate via retry cap
app3 = build_rag_app(FakeLLM(doc="yes", hallu="no", ans="yes"), FakeRetriever(DOCS), FakeWeb())
out3 = app3.invoke({"question": "trigger a retry"})
check("hallucination triggers a second generate", out3["steps"].count("generate") == 2)
check("graph terminates (does not loop forever)", out3["steps"].count("generate") <= 3)

print("\nRESULT:", sum(results), "/", len(results), "passed")
sys.exit(0 if all(results) else 1)
