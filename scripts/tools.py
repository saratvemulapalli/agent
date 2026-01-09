from strands import tool
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
        filename = "knowledge/opensearch_semantic_search_guide.md"
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
        filename = "knowledge/dense_vector_models.md"
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
        filename = "knowledge/sparse_vector_models.md"
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
    print("\n[Tool: get_sample_doc] To better understand your requirements, I'd like to see a sample of your data.")
    print("1. Use a mock sample (Randomly selected from common use cases)")
    print("2. Enter a sample document manually")
    
    while True:
        try:
            choice = input("Please choose (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except EOFError:
            return "User cancelled input."

    if choice == '1':
        # Mock samples
        samples = [
            # Case 1: English Content
            """{"content": "The quick brown fox jumps over the lazy dog. This is a sample document for testing search capabilities."}""",
            # Case 2: Chinese Content
            """{"content": "这是一个测试文档。自然语言处理是人工智能领域的一个重要方向，涉及计算机与人类语言的交互。"}"""
        ]
        selected = random.choice(samples)
        print(f"\n[Tool] Selected mock sample:\n{selected}\n")
        return selected
    else:
        print("\nPlease paste your sample document (one line):")
        try:
            user_doc = input("> ")
            return user_doc
        except EOFError:
            return "User cancelled input."
