"""
Runs a RAG query against the FAISS index.
Usage:
    python code/rag_pipeline.py --q "What is X?"
"""

import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from prompt_templates import RAG_CHAT_PROMPT

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "embeddings" / "faiss_store"
CONFIG_PATH = ROOT / "config" / "config.json"

def load_config():
    load_dotenv()
    cfg = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "CHAT_MODEL": "gpt-4o-mini",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "TOP_K_RESULTS": 5,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg

def as_context(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{i}] Source: {src}\n{d.page_content}\n")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--query", dest="query", required=True, help="Natural language question")
    args = parser.parse_args()

    cfg = load_config()
    api_key = cfg.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or config/config.json")

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. Run 'python code/index_documents.py' first."
        )

    embeddings = OpenAIEmbeddings(model=cfg.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    vectordb = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": int(cfg.get("TOP_K_RESULTS", 5))})
    docs = retriever.get_relevant_documents(args.query)

    context = as_context(docs)
    prompt = RAG_CHAT_PROMPT.format(context=context, question=args.query)

    llm = ChatOpenAI(model=cfg.get("CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    response = llm.invoke(prompt)

    print("\n=== Answer ===\n")
    print(response.content.strip())
    print("\n=== Sources ===")
    for d in docs:
        print("-", d.metadata.get("source", "unknown"))

if __name__ == "__main__":
    main()
