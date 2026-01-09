from strands import tool

@tool
def read_knowledge_base() -> str:
    """Read the OpenSearch Semantic Search Guide to retrieve detailed information about search methods.

    Returns:
        str: The content of the guide covering BM25, Dense Vector, Sparse Vector, Hybrid, algorithms (HNSW, IVF, etc.), cost profiles, and deployment options.
    """
    try:
        # Assuming the file is in the same directory or accessible via relative path
        # Since this script is in scripts/ folder, we need to go one level up if run from there, 
        # or if run from root (as module), it depends on CWD.
        # But typically we run from root.
        with open("opensearch_semantic_search_guide.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading knowledge base: {e}"
