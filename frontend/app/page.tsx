"use client";

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

type Source = { n: number; source: string; preview: string };
type Msg = {
  role: "user" | "assistant";
  content: string;
  steps?: string[];
  sources?: Source[];
};

const STEP_LABEL: Record<string, string> = {
  retrieve: "Retrieve",
  grade_documents: "Grade docs",
  web_search: "Web search",
  generate: "Generate",
};

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function ask() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setBusy(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer || "(no answer)",
          steps: data.steps || [],
          sources: data.sources || [],
        },
      ]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ Could not reach the API at ${API}. ${e?.message || ""}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function ingest(files: FileList | null) {
    if (!files || files.length === 0) return;
    setStatus(`Uploading ${files.length} file(s)…`);
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    try {
      const res = await fetch(`${API}/ingest`, { method: "POST", body: fd });
      const d = await res.json();
      setStatus(
        `Indexed ${d.chunks} chunks from ${d.files} file(s)` +
          (d.images_captioned ? ` · ${d.images_captioned} image(s) captioned` : "")
      );
    } catch (e: any) {
      setStatus(`Upload failed: ${e?.message || e}`);
    }
  }

  return (
    <div className="wrap">
      <div className="header">
        <div className="logo">🧠</div>
        <div>
          <h1>Adaptive RAG Assistant</h1>
        </div>
      </div>
      <p className="sub">
        Ask about your documents. The agent retrieves, grades its own evidence, falls back to web
        search when needed, and cites sources.
      </p>

      <div className="uploader">
        <span className="grow">📎 Add PDFs or text files to the knowledge base (images inside PDFs are read too)</span>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          onChange={(e) => ingest(e.target.files)}
        />
        <button className="btn ghost" onClick={() => fileRef.current?.click()}>
          Upload
        </button>
      </div>
      {status && <div className="status">{status}</div>}

      <div className="chat">
        {messages.length === 0 && (
          <div className="empty">Upload a document, then ask a question to begin.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.role === "assistant" && m.steps && m.steps.length > 0 && (
                <div className="steps">
                  {m.steps.map((s, j) => (
                    <span key={j} className={`step ${s}`}>
                      {STEP_LABEL[s] || s}
                    </span>
                  ))}
                </div>
              )}
              {m.content}
              {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                <div className="sources">
                  <div className="t">Sources</div>
                  {m.sources.map((s) => (
                    <div key={s.n} className="src">
                      <b>[{s.n}] {s.source}</b> — {s.preview}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="msg assistant">
            <div className="bubble">
              <span className="dots">Thinking</span>
            </div>
          </div>
        )}
      </div>

      <div className="composer">
        <div className="inner">
          <input
            value={input}
            placeholder="Ask a question about your documents…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            disabled={busy}
          />
          <button className="btn primary" onClick={ask} disabled={busy || !input.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
