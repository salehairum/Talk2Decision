from filter_for_search import SentenceTransformerEmbedder, build_index
from query import run_query
from loader import load_slack_export
from preprocess import preprocess_messages

# ================================================================
# Entrypoint function
# ================================================================
def entrypoint(slack_export_path: str, top_k: int = 5):
    """
    Loads Slack export, preprocesses messages, builds index, and
    starts an interactive prompt for running multiple queries.
    
    Args:
        slack_export_path: Path to Slack JSON export file.
        top_k: Number of top results to return for each query.
    """
    print("Preparing the system...")
    
    # 1) Load and preprocess
    raw_messages = load_slack_export(slack_export_path)
    clean_messages = preprocess_messages(raw_messages)

    # 2) Initialize embedder and build index
    embedder = SentenceTransformerEmbedder()
    index = build_index(clean_messages, embedder=embedder)

    print("System ready. You can now enter queries (type 'exit' to quit).\n")

    # 3) Interactive query loop
    while True:
        query = input("Query > ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not query:
            continue

        print(f"\n-- Hybrid search results for: '{query}' --")
        results = run_query(index, query, top_k=top_k, embedder=embedder)

        for i, res in enumerate(results):
            msg = res["message"]
            sources = ", ".join(res["source"])
            print(f"[{i+1}] score={res['score']} source={sources}")
            print(f"    @{msg.get('author_name', '')} {msg.get('timestamp', '')}")
            print(f"    {msg.get('content_clean', '')[:120]}")

            if res.get("window"):
                win = res["window"]
                print(f"    [WINDOW {win['start']}:{win['end']}] {win['text'][:120]}")

        print("-" * 60)


# ================================================================
# Run entrypoint if executed as main
# ================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <slack_export.json>")
        sys.exit(1)

    slack_file = sys.argv[1]
    entrypoint(slack_file, top_k=5)