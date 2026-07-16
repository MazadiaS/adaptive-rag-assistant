import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Adaptive RAG Assistant",
  description: "Agentic RAG over your documents — retrieve, self-grade, web-fallback, cite.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
