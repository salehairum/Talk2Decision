# semantic search and regular search
"""
filter.py
---------
Two search modes over preprocessed T2D messages:

1) Keyword search over a persisted inverted index
2) Semantic search over a persisted Chroma vector index

"""

from __future__ import annotations

import math
from collections import defaultdict
from embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    SearchIndex,
    build_index,
    _tokenise,
    _get_chroma_collection,
    _get_windows_collection,
)

# ================================================================
# Retrieval
# ================================================================


def keyword_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    n_docs = len(index.messages)
    scores: dict[int, float] = defaultdict(float)


    for token in query_tokens:
        postings = index.inverted_index.get(token, [])
        if not postings:
            continue
        idf = math.log((n_docs + 1) / (len(postings) + 1)) + 1.0
        for pos in postings:
            scores[pos] += idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "message": index.messages[pos],
            "score": round(score, 4),
            "rank": i + 1,
            "position": pos,
        }
        for i, (pos, score) in enumerate(ranked)
    ]


def semantic_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    """
    Semantic retrieval across BOTH:
    - message-level embeddings
    - window-level embeddings
    Thread info is preserved for later aggregation.
    """
    embedder = embedder or SentenceTransformerEmbedder()

    msg_collection = _get_chroma_collection(reset=False)
    win_collection = _get_windows_collection(reset=False)

    if msg_collection.count() == 0 and win_collection.count() == 0:
        raise ValueError("No embeddings found. Run build_index first.")

    query_vec = embedder.embed([query])[0]
    fetch_k = top_k * 2

    combined: list[dict] = []

    # ---------------- Message embeddings ----------------
    msg_results = msg_collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        include=["distances", "metadatas", "documents"],
    )
    msg_ids = msg_results.get("ids", [[]])[0]
    msg_distances = msg_results.get("distances", [[]])[0]

    for i, doc_id in enumerate(msg_ids):
        pos = int(doc_id)
        if not (0 <= pos < len(index.messages)):
            continue
        message = index.messages[pos]
        score = 1.0 - float(msg_distances[i]) if i < len(msg_distances) else 0.0
        thread_ts = message.get("thread_ts", message.get("timestamp"))

        combined.append({
            "message": message,
            "score": score,
            "position": pos,
            "source": "message",
            "thread_ts": thread_ts,
        })

    # ---------------- Window embeddings ----------------
    win_results = win_collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        include=["distances", "metadatas", "documents"],
    )
    win_ids = win_results.get("ids", [[]])[0]
    win_distances = win_results.get("distances", [[]])[0]
    win_metadatas = win_results.get("metadatas", [[]])[0]
    win_documents = win_results.get("documents", [[]])[0]

    for i, win_id in enumerate(win_ids):
        metadata = win_metadatas[i] if i < len(win_metadatas) else {}
        start = int(metadata.get("start", -1))
        end = int(metadata.get("end", -1))
        if start == -1 or start >= len(index.messages):
            continue
        message = index.messages[start]
        score = 1.0 - float(win_distances[i]) if i < len(win_distances) else 0.0
        thread_ts = message.get("thread_ts", message.get("timestamp"))

        combined.append({
            "message": message,
            "score": score,
            "position": start,
            "source": "window",
            "thread_ts": thread_ts,
            "window": {
                "start": start,
                "end": end,
                "text": win_documents[i] if i < len(win_documents) else "",
            },
        })

    # ---------------- Merge + deduplicate ----------------
    combined.sort(key=lambda x: x["score"], reverse=True)
    seen: dict[int, dict] = {}
    for item in combined:
        pos = item["position"]
        if pos not in seen or item["score"] > seen[pos]["score"]:
            seen[pos] = item

    ranked = list(seen.values())[:top_k]

    # ---------------- Format output ----------------
    output = []
    for i, item in enumerate(ranked):
        output.append({
            "message": item["message"],
            "score": round(item["score"], 4),
            "rank": i + 1,
            "position": item["position"],
            "source": item["source"],
            "thread_ts": item["thread_ts"],
            "window": item.get("window"),
        })

    return output


def hybrid_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
    rrf_k: int = 60,
    root_weight: float = 2.0,
    reply_weight: float = 1.0,
) -> list[dict]:
    """
    Thread-aware hybrid search.
    - Combines keyword + semantic scores via RRF.
    - Aggregates results per thread if message belongs to a thread.
    """
    kw_results = keyword_search(index, query, top_k=top_k * 2)
    sem_results = semantic_search(index, query, top_k=top_k * 2, embedder=embedder)

    # ---------------- Compute RRF per message ----------------
    rrf_scores: dict[int, float] = defaultdict(float)
    sources: dict[int, set[str]] = defaultdict(set)
    windows_data: dict[int, dict] = {}

    for result in kw_results:
        pos = result["position"]
        rrf_scores[pos] += 1.0 / (rrf_k + result["rank"])
        sources[pos].add("keyword")

    for result in sem_results:
        pos = result["position"]
        rrf_scores[pos] += 1.0 / (rrf_k + result["rank"])
        sources[pos].add(result.get("source", "semantic"))
        if result.get("window"):
            windows_data[pos] = result["window"]

    # ---------------- Compute thread-level scores ----------------
    thread_scores: dict[str, float] = defaultdict(float)
    for pos, score in rrf_scores.items():
        msg = index.messages[pos]
        thread_ts = msg.get("thread_ts", msg.get("timestamp"))
        weight = root_weight if msg.get("timestamp") == thread_ts else reply_weight
        thread_scores[thread_ts] += score * weight
        is_threaded=msg.get("is_threaded", False)
        if is_threaded:
            sources[pos].add("thread")

    # ---------------- Rank messages ----------------
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    output = []
    for i, (pos, score) in enumerate(ranked):
        msg = index.messages[pos]
        thread_ts = msg.get("thread_ts", msg.get("timestamp"))
        output.append({
            "message": msg,
            "score": round(score, 6),
            "rank": i + 1,
            "position": pos,
            "source": sorted(list(sources[pos])),
            "window": windows_data.get(pos),
            "thread_score": round(thread_scores[thread_ts], 6)
                            if thread_ts != msg.get("timestamp") else None
        })

    return output

# ================================================================
# Quick sanity check
# ================================================================

if __name__ == "__main__":
    import sys

    from loader import load_slack_export
    from preprocess import preprocess_messages

    if len(sys.argv) < 2:
        print("Usage: python filter.py <slack_export.json>")
        sys.exit(1)

    filepath = sys.argv[1]

    raw = load_slack_export(filepath)
    clean = preprocess_messages(raw)

    embedder = SentenceTransformerEmbedder()
    index = build_index(clean, embedder=embedder)

    print("\n=== Interactive Search ===")
    print("Type your query (or 'exit' to quit)\n")

    while True:
        query = input("Query > ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query:
            continue

        print(f"\n-- Hybrid search: '{query}' --")

        results = hybrid_search(index, query, top_k=5, embedder=embedder)

        for result in results:
            msg = result["message"]
            source = ", ".join(result["source"])
            window = result.get("window")

            print(f"[{result['rank']}] score={result['score']} source={source}")
            print(f"    @{msg.get('author_name', '')} {msg.get('timestamp', '')}")
            print(f"    {msg.get('content_clean', '')[:120]}")

            if window:
                print(f"    [WINDOW {window['start']}:{window['end']}]")
                print(f"    {window['text'][:120]}")

        print("-" * 60)
