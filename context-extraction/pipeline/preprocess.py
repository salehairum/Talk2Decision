#remove emojis, lower case etc

"""
preprocess.py
-------------
Cleans and normalises the `content` field of T2D messages that have
already been loaded by loader.py.

Pipeline (in order):
    1. Slack mention resolver   <@UID> / <@UID|name>  →  @Name
    2. URL stripper             <http://...>  →  removed
    3. Slack formatting         *bold*, _italic_, `code`  →  plain text
    4. Emoji remover            unicode emoji  →  removed
    5. Boilerplate filter       sign-offs, filler turns  →  message dropped
    6. Lowercase
    7. Whitespace normalisation collapse runs of spaces/newlines

Usage:
    from loader import load_slack_export
    from preprocess import preprocess_messages

    raw = load_slack_export("general.json", thread_id="berlin-london")
    clean = preprocess_messages(raw)
"""

import re
import unicodedata
from copy import deepcopy


# ── Slack-specific patterns ────────────────────────────────────────────────

# <@U0AMZS4QFPS> or <@U0AMZS4QFPS|saleha>
_SLACK_MENTION = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]+))?>")

# <http://example.com> or <http://example.com|label>
_SLACK_URL     = re.compile(r"<https?://[^>]+>")

# <!channel>, <!here>, <!everyone>
_SLACK_SPECIAL = re.compile(r"<!(?:channel|here|everyone|subteam\^[^>]+)>")

# Slack markdown: *bold*, _italic_, ~strike~, `inline code`, ```code block```
_SLACK_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_SLACK_INLINE_FMT = re.compile(r"[*_~`]")


# ── Boilerplate detection ──────────────────────────────────────────────────

# If a message matches one of these patterns after cleaning, drop it entirely.
# These are filler turns that add zero signal to a decision.
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(ok|okay|sure|got it|noted|thanks|thank you|thx|np|no problem|sounds good|alright|will do|done|oki|okie|k|kk)[\s!.]*$",
        r"^(yes|no|yep|nope|yup|nah|hmm+|hm+|lol|haha+|😂|👍|👎|✅)[\s!.]*$",
        r"^(best regards|regards|cheers|sincerely|yours truly|warm regards).*",
        r"^\+1$",
    ]
]


# ── Emoji removal ──────────────────────────────────────────────────────────

def _remove_emoji(text: str) -> str:
    """
    Remove unicode emoji characters from text.
    Uses unicodedata category — no external library needed.
    Categories removed: So (Symbol, Other) covers most emoji.
    Also strips variation selectors (U+FE0F) and zero-width joiners (U+200D).
    """
    result = []
    for char in text:
        cp = ord(char)
        # Variation selector, ZWJ, combining enclosing keycap
        if cp in (0xFE0F, 0x200D, 0x20E3):
            continue
        # Misc symbols, dingbats, emoticons, transport/map, supplemental symbols
        if unicodedata.category(char) in ("So", "Cs"):
            continue
        # Emoji ranges not caught by category alone
        if (0x1F600 <= cp <= 0x1F64F   # emoticons
                or 0x1F300 <= cp <= 0x1F5FF  # misc symbols & pictographs
                or 0x1F680 <= cp <= 0x1F6FF  # transport & map
                or 0x1F700 <= cp <= 0x1F77F  # alchemical symbols
                or 0x1F780 <= cp <= 0x1F7FF
                or 0x1F800 <= cp <= 0x1F8FF
                or 0x1F900 <= cp <= 0x1F9FF
                or 0x1FA00 <= cp <= 0x1FAFF
                or 0x2600  <= cp <= 0x26FF   # misc symbols
                or 0x2700  <= cp <= 0x27BF): # dingbats
            continue
        result.append(char)
    return "".join(result)


# ── Punctuation normalisation ──────────────────────────────────────────────

_MULTI_SPACE    = re.compile(r"[ \t]+")
_MULTI_NEWLINE  = re.compile(r"\n{3,}")
_LEADING_PUNCT  = re.compile(r"^[,;:\-–—]+\s*")
_TRAILING_PUNCT = re.compile(r"\s*[,;:\-–—]+$")


def _normalise_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ── Core cleaning function ─────────────────────────────────────────────────

def clean_text(text: str, user_map: dict[str, str] | None = None) -> str:
    """
    Clean a single message string and return the cleaned version.
    Returns an empty string if the message should be dropped.

    Args:
        text:     Raw message content.
        user_map: Optional {user_id: display_name} dict to resolve @mentions.
                  Built automatically by preprocess_messages() from loaded data.
    """
    # 1. Slack code blocks — remove entirely (code ≠ decision text)
    text = _SLACK_CODE_BLOCK.sub("", text)

    # 2. Resolve Slack mentions  <@UID> → @Name  or  @uid if not in map
    def resolve_mention(m):
        uid    = m.group(1)
        inline = m.group(2)           # optional |name part
        if user_map and uid in user_map:
            return f"@{user_map[uid]}"
        if inline:
            return f"@{inline}"
        return f"@{uid}"

    text = _SLACK_MENTION.sub(resolve_mention, text)

    # 3. Strip URLs
    text = _SLACK_URL.sub("", text)

    # 4. Strip special broadcasts
    text = _SLACK_SPECIAL.sub("", text)

    # 5. Strip Slack inline formatting markers
    text = _SLACK_INLINE_FMT.sub("", text)

    # 6. Remove emoji
    text = _remove_emoji(text)

    # 7. Lowercase
    text = text.lower()

    # 8. Normalise whitespace
    text = _normalise_whitespace(text)

    return text


def _is_boilerplate(text: str) -> bool:
    """Return True if the cleaned text is a low-signal filler message."""
    return any(p.match(text) for p in _BOILERPLATE_PATTERNS)


# ── Public API ─────────────────────────────────────────────────────────────

def preprocess_messages(
    messages: list[dict],
    drop_boilerplate: bool = True,
) -> list[dict]:
    """
    Run the full cleaning pipeline over a list of normalised T2D messages.

    Each message dict is expected to have at least a `content` key
    (as produced by loader.py). A new key `content_clean` is added;
    the original `content` is preserved for debugging.

    Args:
        messages:         List of normalised message dicts from loader.py.
        drop_boilerplate: If True, messages that are pure filler are removed.

    Returns:
        List of cleaned message dicts (boilerplate messages excluded).
    """
    # Build a user_id → display_name map from whatever is in the messages
    user_map: dict[str, str] = {}
    for msg in messages:
        uid  = msg.get("author_id", "")
        name = msg.get("author_name", "")
        if uid and name and uid != name:
            user_map[uid] = name

    cleaned = []
    stats = {"total": len(messages), "dropped_boilerplate": 0, "kept": 0}

    for msg in messages:
        result = deepcopy(msg)
        raw    = msg.get("content", "")

        clean = clean_text(raw, user_map=user_map)

        if not clean:
            # Empty after cleaning — drop silently
            continue

        if drop_boilerplate and _is_boilerplate(clean):
            stats["dropped_boilerplate"] += 1
            continue

        result["content_clean"] = clean
        cleaned.append(result)
        stats["kept"] += 1

    print(
        f"[preprocess] {stats['total']} in → "
        f"{stats['kept']} kept, "
        f"{stats['dropped_boilerplate']} boilerplate dropped"
    )
    return cleaned


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    from loader import load_slack_export

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_slack_export.json>")
        sys.exit(1)

    raw   = load_slack_export(sys.argv[1])
    clean = preprocess_messages(raw)

    print(f"\nSample cleaned messages ({min(3, len(clean))}):")
    for msg in clean[:3]:
        print(json.dumps({
            "author_name":   msg["author_name"],
            "timestamp":     msg["timestamp"],
            "content":       msg["content"],
            "content_clean": msg["content_clean"],
        }, indent=2, ensure_ascii=False))