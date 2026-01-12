from strands import Agent, tool
from strands.models import BedrockModel
from scripts.handler import ThinkingCallbackHandler
from scripts.opensearch_ops_tools import create_index
from scripts.tools import retrieve_from_documentation

# -------------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
# OpenSearch Implementation Engineer

You are an expert OpenSearch implementation engineer.
Your goal is to execute the technical plan provided in the context.

## Your Responsibilities
1.  **Analyze the Plan**: specific index settings, mappings, and configurations.
2.  **Execute**: Use the `create_index` tool to create the indices with the correct configurations.
    *   Construct the index body (settings and mappings) based on the plan.
    *   Call `create_index` with the appropriate index name and body.
3.  **Report**: Confirm successful execution.

## Important Rules
1. When using sparse vector search with SEISMIC or ANN, you should use `sparse_vector` field instead of `rank_features` field.
"""

# -------------------------------------------------------------------------
# Worker Execution
# -------------------------------------------------------------------------

@tool
def worker_agent(context: str) -> str:
    """Set up the index and models based on the final plan.
    
    Args:
        context: The detailed technical plan that has been approved by the user.
        
    Returns:
        str: Status of the execution.
    """
    print(f"\n[Worker] Received context for execution:\n{context}")
    
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    try:
        model = BedrockModel(
            model_id=model_id,
            max_tokens=8192,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                }
            }
        )
        
        agent = Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=[create_index, retrieve_from_documentation],
            callback_handler=ThinkingCallbackHandler(output_color="\033[92m") # Green for worker
        )
        
        instruction = f"Here is the approved plan:\n{context}\nPlease implement this plan by creating the necessary indices."
        
        response = agent(instruction)
        return str(response)

    except Exception as e:
        return f"Error executing worker agent: {str(e)}"

SAMPLE_CONTEXT = """
**Corpus Details:**
- Document Size: 10 million documents
- Language: English (mono-lingual)
- Sample Document: Text content similar to "The quick brown fox jumps over the lazy dog. This is a sample document for testing search capabilities."

**Search Architecture:**
- Hybrid Search: Dense Vector (diskANN) + BM25 keyword search
- Score Normalization: Min-Max or L2 normalization
- Hybrid Weighting: Start with 0.5/0.5, tune toward 0.6-0.7 for dense vector emphasis

**DiskANN Vector Configuration:**
- Engine: faiss
- Method: hnsw
- Mode: on_disk (Binary Quantization + disk re-ranking)
- Parameters for Maximum Accuracy:
  - ef_construction: 512-1000
  - ef_search: 200-500
  - m: 48-64
- Storage: Fast NVMe SSDs for optimal disk re-ranking performance

**Embedding Model:**
- Deployment: SageMaker GPU Endpoint
- Model Options:
  - Primary: intfloat/e5-large-v2 (1024 dimensions, state-of-the-art English)
  - Alternative: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
- Inference: GPU-based for 5-20ms latency

**BM25 Configuration:**
- Standard inverted index with English analyzer

**User Priorities:**
- Best accuracy possible
- No budget constraints
- User explicitly chose diskANN despite accuracy trade-offs

Please proceed with index and model setup.
"""

SAMPLE_CONTEXT_2 = """
- Retrieval Method: Hybrid Search (Dense Vector + Sparse Vector)
        - Dense Vector Configuration:
          * Algorithm: HNSW (lucene or faiss engine)
          * Model: amazon.titan-embed-text-v2 (1536 dimensions)
          * Deployment: Amazon Bedrock API
        - Sparse Vector Configuration:
          * Mode: Doc-only
          * Ingestion Model: amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-gte
          * Ingestion Deployment: SageMaker GPU Endpoint (ml.g5.xlarge)
          * Query Tokenizer: amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1
          * Query Deployment: OpenSearch Node (CPU)
          * Index Backend: rank_features (exact search)
        - Score Normalization: Min-Max or L2 (default 50/50 weight split)
        - No BM25 component (redundant with sparse vector)
"""

SAMPLE_CONTEXT_3 = """
- **Retrieval Method:** Hybrid Search (Dense Vector + Sparse Vector Neural Sparse)
        - **Dense Vector Configuration:**
          - Algorithm: HNSW (Hierarchical Navigable Small World)
          - Model Deployment: Amazon Bedrock Embedding API
          - Model: amazon.titan-embed-text-v2 (1536 dimensions)
        - **Sparse Vector Configuration:**
          - Mode: Doc-Only (for optimal query latency)
          - Index Backend: SEISMIC (ANN for sparse vectors)
          - Ingestion Model: amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-gte (deployed on SageMaker GPU, ml.g5.xlarge)
          - Search Model: amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1 (deployed on OpenSearch Nodes CPU)
        - **Score Normalization:** OpenSearch hybrid query with normalization (min-max or L2)
"""

if __name__ == "__main__":
    print(worker_agent(SAMPLE_CONTEXT_2))
