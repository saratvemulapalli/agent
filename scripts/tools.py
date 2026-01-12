from strands import tool, Agent
from strands_tools import retrieve
import random

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
        filename = "scripts/knowledge/opensearch_semantic_search_guide.md"
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading knowledge base: {e}"

@tool
def read_dense_vector_models() -> str:
    """Read the Dense Vector Models Guide to retrieve available models for Dense Vector Search.

    Returns:
        str: The content of the guide covering models for OpenSearch Node, SageMaker GPU, and External API services.
    """
    try:
        filename = "scripts/knowledge/dense_vector_models.md"
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading dense vector models guide: {e}"

@tool
def read_sparse_vector_models() -> str:
    """Read the Sparse Vector Models Guide to retrieve available models for Sparse Vector Search.

    Returns:
        str: The content of the guide covering models for Doc-Only and Bi-Encoder modes.
    """
    try:
        filename = "scripts/knowledge/sparse_vector_models.md"
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading sparse vector models guide: {e}"

@tool
def get_sample_doc() -> str:
    """Get a sample document from the user database to understand their data structure.
    
    Returns:
        str: The sample document content.
    """

    return """{"content": "The quick brown fox jumps over the lazy dog. This is a sample document for testing search capabilities."}"""

@tool
def retrieve_from_documentation(query: str, numberOfResults: int = 5, score: float = 0.4) -> str:
    """Search from the OpenSearch documentation.

    example query: "sparse_vector field parameters"

    Args:
        query: The query text to search for.
        numberOfResults: The number of results to return.
        score: Minimum score threshold.

    Returns:
        str: The results from the search.
    """
    try:
        # use red font to print the query
        print(f"\033[91m[retrieve_from_documentation] Query: {query}\033[0m")

        # Create a dummy agent to access the retrieve tool functionality
        # The retrieve object from strands_tools needs to be attached to an agent
        agent = Agent(tools=[retrieve])
        
        results = agent.tool.retrieve(
            text=query,
            numberOfResults=numberOfResults,
            knowledgeBaseId="LWG7KNS0UK",
            region="us-east-1",
            score=score,
        )
        return str(results)
    except Exception as e:
        return f"Error retrieving documentation: {e}"
