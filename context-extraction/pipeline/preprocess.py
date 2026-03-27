"""
preprocess.py
-------------
Cleans and normalises the `content` field of T2D messages that have
already been loaded by loader.py.

Pipeline (in order):
    1.  Unicode normalisation     NFKC — smart quotes, accents, nbsp, etc.
    2.  Slack code block strip    ```...```  →  removed
    3.  Emoji shortcode handler  :thumbsup:→yes  :fire:→removed  unknown→removed
    4.  Emoji removal             all remaining unicode emoji stripped
    5.  Slack mention resolver    <@UID>  →  @Name
    6.  URL stripper              <http://...>  →  removed
    7.  Slack special strip       <!channel>  <!here>  etc.
    8.  Slack formatting strip    *bold*  _italic_  ~strike~  `code`
    9.  Punctuation normalisation collapse ...  !!  ??  strip decorative chars
    10. Abbreviation expansion    u→you  tbh→to be honest  etc.
    11. Lowercase
    12. Whitespace normalisation  collapse spaces/newlines
    13. Boilerplate filter        pure filler messages dropped

Design note on boilerplate:
    Words like "ok", "okay", "yes", "no", "sure" are intentionally NOT
    dropped — in a decision conversation they can signal agreement or
    rejection of an option and are therefore decision-relevant.
    Only messages with zero possible decision signal are dropped.

Usage:
    from loader import load_slack_export
    from preprocess import preprocess_messages

    raw   = load_slack_export("general.json", thread_id="berlin-london")
    clean = preprocess_messages(raw)
"""

import re
import unicodedata
from copy import deepcopy
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# 1. Slack-specific patterns
# ══════════════════════════════════════════════════════════════════

_SLACK_MENTION    = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]+))?>")
_SLACK_URL        = re.compile(r"<https?://[^>]+>")
_SLACK_SPECIAL    = re.compile(r"<!(?:channel|here|everyone|subteam\^[^>]+)>")
_SLACK_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_SLACK_INLINE_FMT = re.compile(r"[*_~`]")


# ══════════════════════════════════════════════════════════════════
# 2. Slack emoji shortcode handler
#
#    regex detects all :word: patterns
#      ↓
#    DECISION dict match → replace with plain English token (yes/no/confirmed/rejected)
#      ↓
#    everything else (noise, unknown) → removed entirely
#
#    e.g.  'Done :+1:'             →  'done yes'
#          'bad idea :thumbsdown:' →  'bad idea no'
#          'great :fire:'          →  'great'   (dropped)
#          ':skull_and_crossbones:'→  ''        (dropped)
# ══════════════════════════════════════════════════════════════════

_SHORTCODE_RE = re.compile(r':[a-z0-9_+\-]+'  r':')
BASE_DIR = Path(__file__).resolve().parent
EMOJI_FILE = BASE_DIR / "emojis.json"
# Load emoji dictionary from JSON
with open(EMOJI_FILE, "r", encoding="utf-8") as f:
    _EMOJI_DATA = json.load(f)
    
# Extract only the emoji names for fast lookup
EMOJI_NAMES = set(_EMOJI_DATA["EMOJIS"].keys())
# _DECISION_SHORTCODES: dict[str, str] = {
#     ":+1:":                          "yes",
#     ":thumbsup:":                    "yes",
#     ":-1:":                          "no",
#     ":thumbsdown:":                  "no",
#     ":white_check_mark:":            "confirmed",
#     ":heavy_check_mark:":            "confirmed",
#     ":ballot_box_with_check:":       "confirmed",
#     ":x:":                           "rejected",
#     ":negative_squared_cross_mark:": "rejected",
#     ":no_entry:":                    "rejected",
#     ":no_entry_sign:":               "rejected",
# }


# def _handle_shortcodes(text: str) -> str:
#     def replace(match: re.Match) -> str:
#         shortcode = match.group(0)
#         token = _DECISION_SHORTCODES.get(shortcode)
#         if token:
#             return f" {token} "
#         return " "
#     return _SHORTCODE_RE.sub(replace, text)
def _handle_shortcodes(text: str) -> str:
    def replace(match: re.Match) -> str:
        shortcode = match.group(0)      # ":coffee:"
        name = shortcode[1:-1]          # "coffee"

        if name in EMOJI_NAMES:
            return " "                  # remove emoji
        return shortcode                # keep if not known

    return _SHORTCODE_RE.sub(replace, text)



# ══════════════════════════════════════════════════════════════════
# 3. Emoji removal  (everything not already converted above)
# ══════════════════════════════════════════════════════════════════

def _remove_emoji(text: str) -> str:
    result = []
    for char in text:
        cp = ord(char)
        if cp in (0xFE0F, 0x200D, 0x20E3):   # variation selector, ZWJ, keycap
            continue
        if unicodedata.category(char) in ("So", "Cs"):
            continue
        if (0x1F600 <= cp <= 0x1F64F
                or 0x1F300 <= cp <= 0x1F5FF
                or 0x1F680 <= cp <= 0x1F6FF
                or 0x1F700 <= cp <= 0x1F77F
                or 0x1F780 <= cp <= 0x1F7FF
                or 0x1F800 <= cp <= 0x1F8FF
                or 0x1F900 <= cp <= 0x1F9FF
                or 0x1FA00 <= cp <= 0x1FAFF
                or 0x2600  <= cp <= 0x26FF
                or 0x2700  <= cp <= 0x27BF):
            continue
        result.append(char)
    return "".join(result)


# ══════════════════════════════════════════════════════════════════
# 4. Punctuation normalisation
# ══════════════════════════════════════════════════════════════════

# Collapse repeated punctuation: "wait..." → "wait.", "really??" → "really?"
_REPEATED_DOTS  = re.compile(r"\.{2,}")
_REPEATED_EXCL  = re.compile(r"!{2,}")
_REPEATED_QUEST = re.compile(r"\?{2,}")
_REPEATED_DASH  = re.compile(r"-{2,}")

# Strip characters that are purely decorative / non-linguistic
# Keeps: letters, digits, basic punctuation (.,!?:;'"-), @, #, /,  newline
_DECORATIVE     = re.compile(r"[^\w\s.,!?:;'\"\-@#+/\n]")

# Whitespace helpers
_MULTI_SPACE    = re.compile(r"[ \t]+")
_MULTI_NEWLINE  = re.compile(r"\n{3,}")


def _normalise_punctuation(text: str) -> str:
    text = _REPEATED_DOTS.sub(".",  text)
    text = _REPEATED_EXCL.sub("!",  text)
    text = _REPEATED_QUEST.sub("?", text)
    text = _REPEATED_DASH.sub("-",  text)
    text = _DECORATIVE.sub(" ",     text)
    return text


def _normalise_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ",    text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════
# 5. Abbreviation expansion
#    Kept intentionally small — only abbreviations that genuinely
#    obscure meaning in a decision context.
#    Applied as whole-word replacements (word boundary anchored).
# ══════════════════════════════════════════════════════════════════

_ABBREVIATIONS: dict[str, str] = {
    r"\bu\b":    "you",
    r"\bur\b":   "your",
    r"\burself\b":"yourself",
    r"\br\b":    "are",
    r"\bidk\b":  "i don't know",
    r"\bidts\b":  "i don't think so",
    r"\bimo\b":  "in my opinion",
    r"\bimho\b": "in my opinion",
    r"\btbh\b":  "to be honest",
    r"\bjbh\b":  "just being honest",
    r"\bngl\b":  "not going to lie",
    r"\basap\b": "as soon as possible",
    r"\bvs\b":   "versus",
    r"\bw/o\b":  "without",
    r"\bw/\b":   "with",
    r"\bbtw\b":  "by the way",
    r"\bfyi\b":  "for your information",
    r"\bafaik\b":"as far as i know",
    r"\bwrt\b":  "with regard to",
    r"\bpls\b":  "please",
    r"\bplej\b":  "please",
    r"\bplz\b":  "please",
    r"\bplss\b":  "please",
    r"\bplejj\b":  "please",
    r"\bplzz\b":  "please",
    r"\bbc\b":   "because",
    r"\bbcz\b":  "because",
    r"\bcuz\b":  "because",
}

# Pre-compile all patterns once
_ABBREV_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _ABBREVIATIONS.items()
]

def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ══════════════════════════════════════════════════════════════════
# 6. Boilerplate detection
#    Only drop messages with ZERO possible decision signal.
#    Notably absent: ok, okay, yes, no, sure, sounds good, alright —
#    these can all indicate agreement/rejection of an option.
# ══════════════════════════════════════════════════════════════════

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # Pure social filler with no decision implication
        r"^(thanks|thank you|thx|ty|tysm)[\s!.]*$",
        r"^(np|no problem|no worries|yw|you're welcome)[\s!.]*$",
        r"^(lol|lmao|lmfao|haha+|hehe+|hihi+|xd)[\s!.]*$",
        r"^(hmm+|hm+|uhh+|umm+|ugh+)[\s!.]*$",
        r"^(kk|k|got it|noted|will do|done|oki|okie)[\s!.]*$",
        r"^(best regards|regards|cheers|sincerely|yours truly|warm regards).*",
        r"^\+1[\s!.]*$",
        # Standalone laugh reactions
        r"^(😂+|🤣+)$",
    ]
]

def _is_boilerplate(text: str) -> bool:
    """Return True only if the message is pure filler with no decision value."""
    return any(p.match(text) for p in _BOILERPLATE_PATTERNS)


# ══════════════════════════════════════════════════════════════════
# 7. Core clean_text function
# ══════════════════════════════════════════════════════════════════

def clean_text(text: str, user_map: dict[str, str] | None = None) -> str:
    """
    Run the full cleaning pipeline on a single message string.

    Args:
        text:     Raw message content from loader.py.
        user_map: {user_id: display_name} for resolving @mentions.

    Returns:
        Cleaned string, or empty string if input was empty.
        (Boilerplate check is done in preprocess_messages, not here,
         so callers get the cleaned text regardless.)
    """
    # Step 0 — Emoji shortcode handler: decision → token, everything else → removed
    text = _handle_shortcodes(text)

    # Step 1 — Unicode normalisation (smart quotes, accents, nbsp, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Step 1b — Replace typographic quotes with straight ASCII equivalents
    # NFKC does not convert U+2019 RIGHT SINGLE QUOTATION MARK → apostrophe
    text = text.replace('‘', "'").replace('’', "'")   # ' '
    text = text.replace('“', '"').replace('”', '"')   # " "
    text = text.replace('′', "'").replace('″', '"')   # prime marks

    # Step 2 — Remove Slack code blocks (code ≠ decision text)
    text = _SLACK_CODE_BLOCK.sub("", text)

    # Step 3 — Remove remaining unicode emoji (shortcodes already handled above)
    text = _remove_emoji(text)

    # Step 5 — Resolve Slack @mentions
    def resolve_mention(m):
        uid    = m.group(1)
        inline = m.group(2)
        if user_map and uid in user_map:
            return f"@{user_map[uid]}"
        return f"@{inline}" if inline else f"@{uid}"

    text = _SLACK_MENTION.sub(resolve_mention, text)

    # Step 6 — Strip URLs
    text = _SLACK_URL.sub("", text)

    # Step 7 — Strip Slack special broadcasts
    text = _SLACK_SPECIAL.sub("", text)

    # Step 8 — Strip Slack inline formatting markers
    text = _SLACK_INLINE_FMT.sub("", text)

    # Step 9 — Expand w/ and w/o BEFORE punctuation strips the slash
    # Note: \b doesn't work across / so we use space/start/end as boundary
    text = re.sub(r"(?<![\w])w/o(?![\w])", "without", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![\w])w/(?![\w])",  "with",    text, flags=re.IGNORECASE)

    # Step 10 — Normalise punctuation and strip decorative characters
    text = _normalise_punctuation(text)

    # Step 11 — Expand remaining abbreviations
    text = _expand_abbreviations(text)

    # Step 11 — Lowercase
    text = text.lower()

    # Step 12 — Normalise whitespace
    text = _normalise_whitespace(text)

    return text


# ══════════════════════════════════════════════════════════════════
# 8. Public API
# ══════════════════════════════════════════════════════════════════

def preprocess_messages(
    messages: list[dict],
    drop_boilerplate: bool = True,
) -> list[dict]:
    """
    Run the full cleaning pipeline over a list of normalised T2D messages.

    Adds `content_clean` to each message dict. Original `content` is
    preserved for debugging/auditing.

    Args:
        messages:         Output of loader.load_slack_export().
        drop_boilerplate: Drop pure filler messages when True.

    Returns:
        List of cleaned message dicts, boilerplate removed.
    """
    # Build user_id → display_name map from loaded messages
    user_map: dict[str, str] = {}
    for msg in messages:
        uid  = msg.get("author_id", "")
        name = msg.get("author_name", "")
        if uid and name and uid != name:
            user_map[uid] = name

    cleaned = []
    stats   = {"total": len(messages), "dropped_empty": 0,
               "dropped_boilerplate": 0, "kept": 0}

    for msg in messages:
        result = deepcopy(msg)
        raw    = msg.get("content", "")

        clean = clean_text(raw, user_map=user_map)

        if not clean:
            stats["dropped_empty"] += 1
            continue

        if drop_boilerplate and _is_boilerplate(clean):
            stats["dropped_boilerplate"] += 1
            continue

        result["content_clean"] = clean
        cleaned.append(result)
        stats["kept"] += 1

    print(
        f"[preprocess] {stats['total']} in → "
        f"{stats['kept']} kept | "
        f"{stats['dropped_boilerplate']} boilerplate dropped | "
        f"{stats['dropped_empty']} empty dropped"
    )
    return cleaned


# ══════════════════════════════════════════════════════════════════
# Quick sanity check
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys
    from loader import load_slack_export

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_slack_export.json>")
        sys.exit(1)

    raw   = load_slack_export(sys.argv[1])
    clean = preprocess_messages(raw)

    print(f"\nSample cleaned messages ({max(5, len(clean))}):")
    for msg in clean[:len(clean)]:
        print(json.dumps({
            "author_name":   msg["author_name"],
            "timestamp":     msg["timestamp"],
            "content":       msg["content"],
            "content_clean": msg["content_clean"],
            "thread_ts":     msg["thread_ts"],
        }, indent=2, ensure_ascii=False))