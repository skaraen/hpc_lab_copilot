from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from config import INDEX_DIR, DEFAULT_TOP_K
from llm.openai_client import embed_texts

class RagIndex:
    def __init__(self):
        index_path = INDEX_DIR / "chunks.index"
        meta_path = INDEX_DIR / "chunks_meta.json"

        if not index_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"Index not found. Run `python -m rag.indexing` first to build it."
            )

        self.index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.chunks: List[Dict[str, Any]] = meta["chunks"]
        self.dim = meta["dim"]

        self.n = self.index.ntotal

    def search(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        emb = embed_texts([query])[0]
        emb_vec = np.array([emb], dtype="float32")
        faiss.normalize_L2(emb_vec)
        scores, idxs = self.index.search(emb_vec, k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        max_s = float(scores.max()) if len(scores) > 0 else 1.0
        min_s = float(scores.min()) if len(scores) > 0 else 0.0
        denom = max(max_s - min_s, 1e-6)

        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            norm_score = (float(score) - min_s) / denom
            results.append(
                {
                    **chunk,
                    "score": float(score),
                    "attention_weight": float(norm_score),
                }
            )
        return results

_rag_index: RagIndex | None = None

def get_rag_index() -> RagIndex:
    global _rag_index
    if _rag_index is None:
        _rag_index = RagIndex()
    return _rag_index


def retrieve_chunks_for_query(query: str, k: int = DEFAULT_TOP_K):
    index = get_rag_index()
    return index.search(query, k=k)
