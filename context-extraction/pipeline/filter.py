#semantic search and regualar search
"""
filter.py
---------
Two search modes over preprocessed T2D messages:

    1. Keyword search   — inverted index, fast, exact/partial token match
    2. Semantic search  — embedding-based cosine similarity (pluggable model)

The embedding model is intentionally left as a swappable backend.
For MVP, a `DummyEmbedder` is provided so the pipeline runs without
any API keys. Swap it for `OpenAIEmbedder` or `SentenceTransformerEmbedder`
when you're ready.

Usage:
    from filter import build_index, keyword_search, semantic_search, hybrid_search

    index = build_index(clean_messages)          # call once after preprocessing

    hits  = keyword_search(index, "career growth berlin")
    hits  = semantic_search(index, "which city is better for jobs")
    hits  = hybrid_search(index, "should i move to berlin or london")
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol


# ═══════════════════════════════════════════════════════════════════
# Embedder protocol — swap implementations without changing search code
# ═══════════════════════════════════════════════════════════════════

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DummyEmbedder:
    """
    Zero-dependency placeholder.
    Returns a bag-of-characters frequency vector (NOT useful for real search —
    replace with a real embedder before production).
    """
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 128
            for ch in text.lower():
                idx = ord(ch) % 128
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    """
    Swap-in when you have an OpenAI key.
    pip install openai
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI()
        self._model  = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbedder:
    """
    Local, no API key needed.
    pip install sentence-transformers
    """
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=False).tolist()


# ═══════════════════════════════════════════════════════════════════
# Index dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SearchIndex:
    messages:       list[dict]                           # preprocessed message dicts
    inverted_index: dict[str, list[int]] = field(default_factory=dict)
    # inverted_index: token → [message positions]

    embeddings:     list[list[float]] | None = None
    # embeddings[i] corresponds to messages[i]


# ═══════════════════════════════════════════════════════════════════
# Tokeniser (shared by both search modes)
# ═══════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "so", "yet", "both", "either",
    "this", "that", "these", "those", "just", "also", "then",
}

_TOKENISE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenise(text: str) -> list[str]:
    tokens = _TOKENISE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════
# Build index
# ═══════════════════════════════════════════════════════════════════

def build_index(
    messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
) -> SearchIndex:
    """
    Build a SearchIndex from a list of preprocessed T2D messages.

    Args:
        messages:    Output of preprocess.preprocess_messages().
        embedder:    Embedder instance. Defaults to DummyEmbedder.
                     Pass OpenAIEmbedder() or SentenceTransformerEmbedder() when ready.
        embed_field: Which message field to embed. Defaults to 'content_clean'.

    Returns:
        A SearchIndex ready for keyword_search / semantic_search.
    """
    embedder = embedder or DummyEmbedder()

    # ── Inverted index ─────────────────────────────────────────────
    inv: dict[str, list[int]] = defaultdict(list)
    for pos, msg in enumerate(messages):
        text = msg.get(embed_field) or msg.get("content", "")
        for token in set(_tokenise(text)):   # set → each token indexed once per msg
            inv[token].append(pos)

    # ── Embeddings ─────────────────────────────────────────────────
    texts      = [msg.get(embed_field) or msg.get("content", "") for msg in messages]
    embeddings = embedder.embed(texts) if texts else []

    print(
        f"[filter] Index built: {len(messages)} messages, "
        f"{len(inv)} unique tokens, "
        f"embedding dim={len(embeddings[0]) if embeddings else 0}"
    )
    return SearchIndex(
        messages=messages,
        inverted_index=dict(inv),
        embeddings=embeddings,
    )


# ═══════════════════════════════════════════════════════════════════
# Keyword search  (TF-IDF-lite BM25-inspired scoring)
# ═══════════════════════════════════════════════════════════════════

def keyword_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search messages using the inverted index.
    Scores each candidate by how many query tokens it contains.

    Returns:
        List of result dicts  {"message": ..., "score": float, "rank": int}
        sorted by score descending.
    """
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    n_docs = len(index.messages)
    scores: dict[int, float] = defaultdict(float)

    for token in query_tokens:
        postings = index.inverted_index.get(token, [])
        if not postings:
            continue
        # IDF: rarer token = higher weight
        idf = math.log((n_docs + 1) / (len(postings) + 1)) + 1.0
        for pos in postings:
            scores[pos] += idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"message": index.messages[pos], "score": round(score, 4), "rank": i + 1}
        for i, (pos, score) in enumerate(ranked)
    ]


# ═══════════════════════════════════════════════════════════════════
# Semantic search  (cosine similarity)
# ═══════════════════════════════════════════════════════════════════

def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    """
    Search messages using cosine similarity over embeddings.

    Note: embedder here must be the SAME model used in build_index().
    For convenience, pass the same instance used when building the index.

    Returns:
        List of result dicts  {"message": ..., "score": float, "rank": int}
    """
    if not index.embeddings:
        raise ValueError("Index has no embeddings. Build index with an embedder first.")

    embedder = embedder or DummyEmbedder()
    query_vec = embedder.embed([query])[0]

    scored = [
        (i, _cosine(query_vec, doc_vec))
        for i, doc_vec in enumerate(index.embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    return [
        {"message": index.messages[pos], "score": round(score, 4), "rank": i + 1}
        for i, (pos, score) in enumerate(top)
    ]


# ═══════════════════════════════════════════════════════════════════
# Hybrid search  (Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════════════

def hybrid_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine keyword and semantic results using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_keyword) + 1/(k + rank_semantic)
    Higher is better. rrf_k=60 is the standard default.

    Returns:
        List of result dicts  {"message": ..., "score": float, "rank": int}
    """
    kw_results  = keyword_search(index, query, top_k=top_k * 2)
    sem_results = semantic_search(index, query, top_k=top_k * 2, embedder=embedder)

    rrf_scores: dict[int, float] = defaultdict(float)

    for result in kw_results:
        pos = index.messages.index(result["message"])
        rrf_scores[pos] += 1.0 / (rrf_k + result["rank"])

    for result in sem_results:
        pos = index.messages.index(result["message"])
        rrf_scores[pos] += 1.0 / (rrf_k + result["rank"])

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"message": index.messages[pos], "score": round(score, 6), "rank": i + 1}
        for i, (pos, score) in enumerate(ranked)
    ]


# ═══════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys
    from loader import load_slack_export
    from preprocess import preprocess_messages

    if len(sys.argv) < 3:
        print("Usage: python filter.py <slack_export.json> <query>")
        sys.exit(1)

    filepath = sys.argv[1]
    query    = " ".join(sys.argv[2:])

    raw      = load_slack_export(filepath)
    clean    = preprocess_messages(raw)
    index    = build_index(clean)

    print(f"\n── Hybrid search: '{query}' ──")
    results = hybrid_search(index, query, top_k=5)
    for r in results:
        msg = r["message"]
        print(f"  [{r['rank']}] score={r['score']}  @{msg['author_name']}  {msg['timestamp']}")
        print(f"       {msg['content_clean'][:120]}")