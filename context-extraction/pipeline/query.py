from filter_for_search import hybrid_search, SentenceTransformerEmbedder

# ================================================================
# 1) Query execution function
# ================================================================
def run_query(index, query: str, top_k: int = 5, embedder: SentenceTransformerEmbedder | None = None) -> list[dict]:
    """
    Runs a hybrid search for a single query against a prepared index.
    
    Args:
        index: The prebuilt SearchIndex containing messages and embeddings.
        query: The query string.
        top_k: Number of top results to return.
        embedder: Optional embedder instance to use.
    
    Returns:
        List of dictionaries containing hybrid search results.
    """
    embedder = embedder or SentenceTransformerEmbedder()
    results = hybrid_search(index, query, top_k=top_k, embedder=embedder)
    return results

