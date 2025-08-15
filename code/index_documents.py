"""
Indexes plain-text documents into a FAISS vector store using OpenAI embeddings.
Place .txt files in data/sample_docs/, then run:
    python code/index_documents.py
"""

import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Paths
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "sample_docs"
INDEX_DIR = ROOT / "data" / "embeddings" / "faiss_store"
CONFIG_PATH = ROOT / "config" / "config.json"

def load_config():
    # Load .env first (if present), then JSON config
    load_dotenv()
    cfg = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 120,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg

def read_text_files(path: Path) -> List[Document]:
    docs: List[Document] = []
    for p in path.glob("**/*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(p.relative_to(ROOT))}))
    return docs

def main():
    cfg = load_config()
    api_key = cfg.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or config/config.json")

    raw_docs = read_text_files(DOCS_DIR)
    if not raw_docs:
        print(f"[index] No .txt documents found in {DOCS_DIR}. Add files and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg.get("CHUNK_SIZE", 800)),
        chunk_overlap=int(cfg.get("CHUNK_OVERLAP", 120)),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=cfg.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    vectordb = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(INDEX_DIR))

    print(f"[index] Indexed {len(raw_docs)} files into {len(chunks)} chunks.")
    print(f"[index] FAISS store saved to: {INDEX_DIR}")

if __name__ == "__main__":
    main()
