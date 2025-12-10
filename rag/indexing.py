from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
import tiktoken
from pypdf import PdfReader

from config import (
    DOCS_DIR,
    CODE_DIR,
    LOGS_DIR,
    INDEX_DIR,
    MAX_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from llm.openai_client import embed_texts

encoder = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> List[int]:
    return encoder.encode(text)


def _detokenize(tokens: List[int]) -> str:
    return encoder.decode(tokens)


def _chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    toks = _tokenize(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + max_tokens)
        chunk_tokens = toks[start:end]
        chunks.append(_detokenize(chunk_tokens))
        if end == len(toks):
            break
        start = end - overlap
    return chunks


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def collect_documents() -> List[Dict[str, Any]]:
    docs = []
    for root_dir, dtype in [
        (DOCS_DIR, "doc"),
        (CODE_DIR, "code"),
        (LOGS_DIR, "log"),
    ]:
        if not root_dir.exists():
            continue
        for path in root_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            try:
                text = _read_file(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            docs.append(
                {
                    "id": str(path.relative_to(root_dir)),
                    "path": str(path),
                    "type": dtype,
                    "text": text,
                }
            )
    return docs


def build_indices():
    docs = collect_documents()
    print(f"Collected {len(docs)} documents.")

    all_chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks = _chunk_text(doc["text"], MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        for i, c in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['id']}::chunk_{i}",
                    "doc_id": doc["id"],
                    "source_path": doc["path"],
                    "type": doc["type"],
                    "text": c,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts) 
    emb_matrix = np.array(embeddings, dtype="float32")

    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb_matrix)
    index.add(emb_matrix)

    faiss.write_index(index, str(INDEX_DIR / "chunks.index"))

    meta = {
        "chunks": all_chunks,
        "dim": dim,
    }
    (INDEX_DIR / "chunks_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Index + metadata written to", INDEX_DIR)


if __name__ == "__main__":
    build_indices()
