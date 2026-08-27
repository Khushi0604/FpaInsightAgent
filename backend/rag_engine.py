"""
RAG Engine
- Loads all CSVs from data/
- Converts rows → text chunks
- Embeds with Sentence Transformers
- Stores in FAISS
- Retrieves top-k relevant chunks for a query
"""

import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, free, no API key needed

_index     = None
_chunks    = []
_embedder  = None


def _load_chunks() -> list[str]:
    """Convert every CSV row into a readable text chunk."""
    chunks = []
    for fname in ["mrr.csv", "customers.csv", "cac.csv", "kpi_summary.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        source = fname.replace(".csv", "")
        for _, row in df.iterrows():
            row_text = f"[{source}] " + " | ".join(
                f"{col}: {val}" for col, val in row.items()
            )
            chunks.append(row_text)
    return chunks


def build_index():
    """Build FAISS index from CSV chunks. Called once at startup."""
    global _index, _chunks, _embedder
    print("Building RAG index...")
    _embedder = SentenceTransformer(EMBED_MODEL)
    _chunks   = _load_chunks()
    embeddings = _embedder.encode(_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    dim    = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)   # inner-product = cosine after normalisation
    _index.add(embeddings)
    print(f"  ✓ Indexed {len(_chunks)} chunks from {DATA_DIR}")


def retrieve(query: str, top_k: int = 8) -> str:
    """Return top_k relevant chunks as a single context string."""
    if _index is None:
        build_index()
    q_vec = _embedder.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)
    _, indices = _index.search(q_vec, top_k)
    results = [_chunks[i] for i in indices[0] if i < len(_chunks)]
    return "\n".join(results)
