# semantic search and regular search
"""
embedder.py
---------

This module supports:
- Building both indexes from scratch
- Appending new messages to both existing indexes
- Retrieving semantic results from Chroma (vector index)

Global configuration:
- EMBEDDING_MODEL: default embedding model
- CACHE_DIR: dedicated local model cache directory
- INDEX_DIR: persisted index storage (inverted index + Chroma DB)
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import chromadb

def _group_by_thread(messages: list[dict]):
    threads = defaultdict(list)

    for msg in messages:
        thread_ts = msg.get("thread_ts")

        # fallback: treat message as its own thread
        if not thread_ts:
            thread_ts = msg["ts_raw"]  # unique per message

        threads[thread_ts].append(msg)

    return threads


# ================================================================
# Global configuration
# ================================================================

# Set the embedding model here to change it globally.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Dedicated model cache directory.
CACHE_DIR = Path(__file__).parent.parent / ".model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Persisted index storage.
INDEX_DIR = Path(__file__).parent.parent / ".index_store"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.pkl"
CHROMA_DIR = INDEX_DIR / "chroma_db"
CHROMA_COLLECTION = "messages"
CHROMA_WINDOWS_COLLECTION = "windows"
WINDOW_SIZE = 4
WINDOW_STRIDE = 1


# ================================================================
# Embedder protocol and implementations
# ================================================================

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DummyEmbedder:
    """Zero-dependency placeholder embedder for local testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 128
            for ch in text.lower():
                vec[ord(ch) % 128] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    """Swap-in when you have an OpenAI key."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbedder:
    """
    Local SentenceTransformer embedder.

    Models are loaded from CACHE_DIR if available. If missing, they are downloaded
    from Hugging Face once and persisted in CACHE_DIR.
    """

    def __init__(self, model: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model_name = model or EMBEDDING_MODEL
        print(
            f"[SentenceTransformerEmbedder] Loading '{self._model_name}' "
            f"with cache at {CACHE_DIR}"
        )
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=str(CACHE_DIR),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if "bge" in self._model_name.lower():
            texts = [
                "Represent this sentence for searching relevant passages: " + text
                for text in texts
            ]

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


# ================================================================
# Search index dataclass
# ================================================================

@dataclass
class SearchIndex:
    messages: list[dict]
    inverted_index: dict[str, list[int]] = field(default_factory=dict)
    embeddings: list[list[float]] | None = None


# ================================================================
# Tokenization
# ================================================================

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


def _message_text(msg: dict[str, Any], embed_field: str) -> str:
    return (msg.get(embed_field) or msg.get("content") or "").strip()


# ================================================================
# Persistence helpers
# ================================================================


def _get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

    return client.get_or_create_collection(name=CHROMA_COLLECTION)

def _get_windows_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_WINDOWS_COLLECTION)
        except Exception:
            pass

    return client.get_or_create_collection(name=CHROMA_WINDOWS_COLLECTION)

def save_inverted_index(inv_index: dict[str, list[int]]) -> None:
    with open(INVERTED_INDEX_PATH, "wb") as fh:
        pickle.dump(inv_index, fh)


def load_inverted_index() -> dict[str, list[int]]:
    if not INVERTED_INDEX_PATH.exists():
        return {}
    with open(INVERTED_INDEX_PATH, "rb") as fh:
        data = pickle.load(fh)
    return data if isinstance(data, dict) else {}


def _build_inverted_index(
    messages: list[dict],
    embed_field: str,
    start_id: int = 0,
    existing: dict[str, list[int]] | None = None,
) -> dict[str, list[int]]:
    inv: dict[str, list[int]] = defaultdict(list)

    if existing:
        for token, postings in existing.items():
            inv[token].extend(postings)

    for offset, msg in enumerate(messages):
        pos = start_id + offset
        for token in set(_tokenise(_message_text(msg, embed_field))):
            inv[token].append(pos)

    return dict(inv)


def _next_message_id_from_inverted(inv_index: dict[str, list[int]]) -> int:
    max_id = -1
    for postings in inv_index.values():
        if postings:
            max_id = max(max_id, max(postings))
    return max_id + 1


# ================================================================
# Build and append index flows
# ================================================================


def build_index(
    messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
) -> SearchIndex:
    """
    Build and persist both indexes from scratch.

    - Inverted index is saved to INVERTED_INDEX_PATH
    - Embeddings are stored in Chroma collection CHROMA_COLLECTION
    """
    embedder = embedder or SentenceTransformerEmbedder()

    inv_index = _build_inverted_index(messages=messages, embed_field=embed_field)
    save_inverted_index(inv_index)

    collection = _get_chroma_collection(reset=True)

    texts = [_message_text(msg, embed_field) for msg in messages]
    embeddings = embedder.embed(texts) if texts else []

    if texts:
        ids = [str(i) for i in range(len(messages))]
        metadatas = [
            {
                "author_name": str(msg.get("author_name", "")),
                "timestamp": str(msg.get("timestamp", "")),
                "position": i,
            }
            for i, msg in enumerate(messages)
        ]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(
        f"[filter] Index built: {len(messages)} messages, "
        f"{len(inv_index)} unique tokens, "
        f"vector_count={collection.count()}"
    )

    # after message embeddings are added

    window_count = add_window_embeddings(
        messages=messages,
        embedder=embedder,
        embed_field=embed_field,
        reset=True,
    )

    print(
        f"[filter] Index built: {len(messages)} messages, "
        f"{len(inv_index)} unique tokens, "
        f"vector_count={collection.count()}, "
        f"window_count={window_count}"
    )
    return SearchIndex(
        messages=messages,
        inverted_index=inv_index,
        embeddings=embeddings,
    )

def _build_windows_by_thread(
    messages: list[dict],
    embed_field: str,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
):
    threads = _group_by_thread(messages)

    windows = []

    for thread_ts, thread_msgs in threads.items():
        if len(thread_msgs) == 1:
            # single message → skip (already handled by message embeddings)
            continue

        for i in range(0, len(thread_msgs) - window_size + 1, stride):
            chunk = thread_msgs[i : i + window_size]

            text = " ".join(
                _message_text(m, embed_field)
                for m in chunk
                if _message_text(m, embed_field)
            ).strip()

            if not text:
                continue

            windows.append(
                {
                    "text": text,
                    "start": chunk[0]["ts_raw"],
                    "end": chunk[-1]["ts_raw"],
                    "thread_ts": thread_ts,
                    "type": "thread_window",
                }
            )

    return windows

def _build_pseudo_windows(
    messages: list[dict],
    embed_field: str,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
):
    # messages that are NOT in real threads
    standalone = [
        m for m in messages
        if m.get("thread_ts") is None or m.get("thread_ts") == m.get("ts_raw")
    ]

    windows = []

    for i in range(0, len(standalone) - window_size + 1, stride):
        chunk = standalone[i : i + window_size]

        text = " ".join(
            _message_text(m, embed_field)
            for m in chunk
            if _message_text(m, embed_field)
        ).strip()

        if not text:
            continue

        windows.append(
            {
                "text": text,
                "start": chunk[0]["ts_raw"],
                "end": chunk[-1]["ts_raw"],
                "thread_ts": None,
                "type": "pseudo_window",
            }
        )

    return windows

def add_to_inverted_index(
    new_messages: list[dict],
    embed_field: str = "content_clean",
    start_id: int | None = None,
) -> dict[str, list[int]]:
    """
    Append new messages to the persisted inverted index.

    Returns the updated inverted index dictionary.
    """
    existing = load_inverted_index()
    if start_id is None:
        start_id = _next_message_id_from_inverted(existing)

    updated = _build_inverted_index(
        messages=new_messages,
        embed_field=embed_field,
        start_id=start_id,
        existing=existing,
    )
    save_inverted_index(updated)
    return updated

def add_embeddings(
    new_messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
    start_id: int | None = None,
) -> list[list[float]]:
    """
    Append new embeddings to the persisted Chroma vector index.

    Returns the newly created embedding vectors.
    """
    if not new_messages:
        return []

    embedder = embedder or SentenceTransformerEmbedder()
    collection = _get_chroma_collection(reset=False)

    if start_id is None:
        start_id = collection.count()

    texts = [_message_text(msg, embed_field) for msg in new_messages]
    embeddings = embedder.embed(texts)
    ids = [str(start_id + i) for i in range(len(new_messages))]
    metadatas = [
        {
            "author_name": str(msg.get("author_name", "")),
            "timestamp": str(msg.get("timestamp", "")),
            "position": start_id + i,
        }
        for i, msg in enumerate(new_messages)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return embeddings

def add_window_embeddings(
    messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
    reset: bool = False,
):
    if not messages:
        return 0

    embedder = embedder or SentenceTransformerEmbedder()
    collection = _get_windows_collection(reset=reset)

    thread_windows = _build_windows_by_thread(messages, embed_field)
    pseudo_windows = _build_pseudo_windows(messages, embed_field)

    all_windows = thread_windows + pseudo_windows

    if not all_windows:
        return 0

    texts = [w["text"] for w in all_windows]
    embeddings = embedder.embed(texts)

    ids = [
        f"w_{w['type']}_{w['start']}_{w['end']}"
        for w in all_windows
    ]

    metadatas = [
        {
            "start": w["start"],
            "end": w["end"],
            "thread_ts": w["thread_ts"],
            "type": w["type"],  # IMPORTANT
        }
        for w in all_windows
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(all_windows)

def add_to_index(
    index: SearchIndex,
    new_messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
) -> SearchIndex:
    """
    Append messages to both persisted indexes and in-memory SearchIndex.
    """
    if not new_messages:
        return index

    start_id = len(index.messages)
    updated_inv = add_to_inverted_index(
        new_messages=new_messages,
        embed_field=embed_field,
        start_id=start_id,
    )
    new_embeddings = add_embeddings(
        new_messages=new_messages,
        embedder=embedder,
        embed_field=embed_field,
        start_id=start_id,
    )

    index.messages.extend(new_messages)
    index.inverted_index = updated_inv

    if index.embeddings is None:
        index.embeddings = new_embeddings
    else:
        index.embeddings.extend(new_embeddings)

    add_window_embeddings_incremental(
        all_messages=index.messages,
        new_messages=new_messages,
        embedder=embedder,
        embed_field=embed_field,
    )
    
    return index

def add_window_embeddings_incremental(
    all_messages: list[dict],
    new_messages: list[dict],
    embedder: Embedder | None = None,
    embed_field: str = "content_clean",
):
    """
    Build windows ONLY for new regions (avoid recomputing everything).
    """
    if not new_messages:
        return 0

    embedder = embedder or SentenceTransformerEmbedder()
    collection = _get_windows_collection(reset=False)

    start = len(all_messages) - len(new_messages)

    # include overlap to preserve context
    window_start = max(0, start - WINDOW_SIZE + 1)

    windows = _build_windows(
        all_messages[window_start:],  # slice
        embed_field,
    )

    if not windows:
        return 0

    texts = [w["text"] for w in windows]
    embeddings = embedder.embed(texts)

    ids = [
        f"w_{window_start + w['start']}_{window_start + w['end']}"
        for w in windows
    ]

    metadatas = [
        {
            "start": window_start + w["start"],
            "end": window_start + w["end"],
            "type": "window",
        }
        for w in windows
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(windows)
