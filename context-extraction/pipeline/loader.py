#loads messages
"""
loader.py
---------
Loads a Slack channel export (JSON) and normalises it into the T2D
unified message schema:

    {
        "message_id":  str,   # client_msg_id or generated from ts
        "thread_id":   str,   # Slack channel name (one channel = one decision)
        "role":        str,   # "user" (all human messages for now)
        "author_id":   str,   # Slack user ID  e.g. "U0AMZS4QFPS"
        "author_name": str,   # Real name from user_profile e.g. "Saleha Irum"
        "content":     str,   # Raw text (will be cleaned in preprocess.py)
        "timestamp":   str,   # ISO 8601  e.g. "2024-03-15T10:31:02"
        "ts_raw":      float  # Original Slack float ts, kept for sorting
    }

Usage:
    from loader import load_slack_export
    messages = load_slack_export("path/to/channel.json", thread_id="berlin-london")
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


# Slack message subtypes that carry zero decision-relevant content
_SKIP_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "channel_archive",
    "channel_unarchive",
    "channel_purpose",
    "channel_topic",
    "channel_name",
    "bot_message",       # bot messages filtered here; can relax later
    "slackbot_response",
}


def _ts_to_iso(ts: str) -> str:
    """Convert Slack float timestamp string to ISO 8601 UTC string."""
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _extract_author(msg: dict) -> tuple[str, str]:
    """
    Return (author_id, author_name) from a message dict.
    Falls back gracefully when user_profile is missing.
    """
    author_id = msg.get("user", "UNKNOWN")

    profile = msg.get("user_profile", {})
    # Prefer real_name, then first_name, then the Slack username, then the ID
    author_name = (
        profile.get("real_name")
        or profile.get("first_name")
        or profile.get("name")
        or author_id
    )
    return author_id, author_name.strip()


def _make_message_id(msg: dict) -> str:
    """Use client_msg_id when present, otherwise derive one from ts."""
    return msg.get("client_msg_id") or f"ts-{msg.get('ts', uuid.uuid4().hex)}"


def load_slack_export(filepath: str | Path, thread_id: str | None = None) -> list[dict]:
    """
    Load a Slack channel export JSON file and return a list of normalised
    T2D message dicts, sorted oldest-first.

    Args:
        filepath:  Path to the Slack export .json file.
        thread_id: Logical name for this decision thread.
                   Defaults to the file stem (e.g. "general" from general.json).

    Returns:
        List of normalised message dicts.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Slack export not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_messages = json.load(f)

    if not isinstance(raw_messages, list):
        raise ValueError("Expected a JSON array at the top level of the Slack export.")

    thread_id = thread_id or path.stem
    normalised = []

    for msg in raw_messages:
        # ── Skip non-message types ──────────────────────────────────────────
        if msg.get("type") != "message":
            continue

        # ── Skip noise subtypes ────────────────────────────────────────────
        subtype = msg.get("subtype", "")
        if subtype in _SKIP_SUBTYPES:
            continue

        # ── Skip messages with no usable text ─────────────────────────────
        text = msg.get("text", "").strip()
        if not text:
            continue

        author_id, author_name = _extract_author(msg)
        ts_raw = float(msg.get("ts", 0))

        normalised.append({
            "message_id":  _make_message_id(msg),
            "thread_id":   thread_id,
            "role":        "user",          # extend later for bot/assistant role
            "author_id":   author_id,
            "author_name": author_name,
            "content":     text,            # raw — preprocess.py cleans this
            "timestamp":   _ts_to_iso(msg["ts"]),
            "ts_raw":      ts_raw,
        })

    # Ensure chronological order (Slack exports are usually sorted, but not guaranteed)
    normalised.sort(key=lambda m: m["ts_raw"])

    return normalised


def save_normalised(messages: list[dict], output_path: str | Path) -> None:
    """Save normalised messages to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    print(f"[loader] Saved {len(messages)} messages → {out}")


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_slack_export.json> [thread_id]")
        sys.exit(1)

    filepath = sys.argv[1]
    thread   = sys.argv[2] if len(sys.argv) > 2 else None
    msgs     = load_slack_export(filepath, thread_id=thread)

    print(f"[loader] Loaded {len(msgs)} messages from '{filepath}'")
    if msgs:
        print("[loader] First message:")
        print(json.dumps(msgs[0], indent=2))
        print("[loader] Last message:")
        print(json.dumps(msgs[-1], indent=2))