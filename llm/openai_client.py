from __future__ import annotations
from typing import List, Dict, Any
from openai import OpenAI
from config import DEFAULT_TEXT_MODEL, DEFAULT_EMBEDDING_MODEL

client = OpenAI()


def embed_texts(texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL):
    if not texts:
        return []

    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def generate_response(
    *,
    system_prompt: str,
    user_prompt: str,
    context_chunks: List[Dict[str, Any]] | None = None,
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.2,
) -> str:
    context_text = ""
    if context_chunks:
        lines = []
        for i, c in enumerate(context_chunks):
            src = c.get("source", "unknown")
            score = c.get("score", 0.0)
            lines.append(
                f"[CHUNK {i+1} | score={score:.2f} | source={src}]\n{c['text']}\n"
            )
        context_text = (
            "You have access to the following context chunks:\n\n"
            + "\n\n".join(lines)
            + "\n\nUse these chunks when answering. Cite CHUNK IDs when you rely on them."
        )

    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    if context_text:
        input_messages.append({"role": "system", "content": context_text})
    input_messages.append({"role": "user", "content": user_prompt})

    resp = client.responses.create(
        model=model,
        input=input_messages,
        temperature=temperature,
    )
    return resp.output_text
