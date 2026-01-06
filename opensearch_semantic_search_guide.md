# OpenSearch Semantic Search Methods - Comprehensive Guide

This document provides a detailed comparison of all semantic search methods available in OpenSearch, designed to help recommend the optimal search solution based on user requirements including data scale, language support, cost budget, latency expectations, accuracy needs, and model deployment preferences.

---

## Table of Contents

1. [Overview of Search Methods](#1-overview-of-search-methods)
2. [BM25 (Lexical Search)](#2-bm25-lexical-search)
3. [Dense Vector Search](#3-dense-vector-search)
4. [Sparse Vector Search](#4-sparse-vector-search)
5. [Hybrid Search](#5-hybrid-search)
6. [Model Deployment Options](#6-model-deployment-options)
7. [Decision Matrix](#7-decision-matrix)
8. [Quick Reference Tables](#8-quick-reference-tables)

---

## 1. Overview of Search Methods

| Method | Type | Core Mechanism | Best For |
|--------|------|----------------|----------|
| BM25 | Lexical | Term frequency + inverse document frequency | Exact keyword matching, structured queries |
| Dense Vector | Semantic | Dense embeddings (typically 384-1536 dims) | Semantic similarity, multilingual |
| Sparse Vector | Semantic | Sparse embeddings with learned term weights | Balance of lexical + semantic |
| Hybrid | Combined | Combines multiple methods with score fusion | Maximum recall and precision |

---

## 2. BM25 (Lexical Search)

### 2.1 Overview

BM25 (Best Matching 25) is the default ranking algorithm in OpenSearch. It calculates relevance based on term frequency (TF), inverse document frequency (IDF), and document length normalization.

### 2.2 Accuracy Characteristics

| Aspect | Rating | Notes |
|--------|--------|-------|
| Exact Match Precision | ⭐⭐⭐⭐⭐ | Excellent for exact keyword queries |
| Semantic Understanding | ⭐⭐ | Cannot understand synonyms or paraphrases |
| Out-of-vocabulary Handling | ⭐ | Fails completely on unseen terms |
| Domain-specific Terms | ⭐⭐⭐⭐⭐ | Excellent for technical/domain vocabulary |

**Strengths:**
- Perfect for exact keyword matching
- Handles rare/domain-specific terminology well
- No vocabulary mismatch between query and index

**Weaknesses:**
- Cannot understand semantic meaning
- Fails on synonyms (e.g., "car" vs "automobile")
- Language-dependent (requires language-specific analyzers)

### 2.3 Cost Profile

| Resource | Cost Level | Details |
|----------|------------|---------|
| Storage | ⭐ (Low) | Only inverted index, typically 10-30% of raw text size |
| Memory | ⭐ (Low) | Field data cache only when needed |
| CPU (Indexing) | ⭐ (Low) | Simple tokenization and analysis |
| CPU (Query) | ⭐ (Low) | Efficient inverted index lookup |

**Storage Estimation:**
```
Index Size ≈ Raw Text Size × 0.1 to 0.3
Example: 1GB text → 100-300MB index
```

### 2.4 Latency Characteristics

| Data Scale | Typical Latency | Notes |
|------------|-----------------|-------|
| < 1M docs | 1-10ms | Near-instant |
| 1M-10M docs | 5-50ms | Still very fast |
| 10M-100M docs | 20-200ms | May need optimization |
| > 100M docs | 50-500ms | Consider sharding strategy |

**Scaling Behavior:**
- Latency grows sub-linearly with data size (O(log n) for most queries)
- Horizontal scaling is straightforward
- Query complexity significantly affects latency

### 2.5 Unique Features & Query Types

BM25 supports several special query types that vector search cannot:

| Query Type | Description | Use Case |
|------------|-------------|----------|
| `prefix` | Matches terms starting with specified prefix | Autocomplete, partial matching |
| `wildcard` | Pattern matching with * and ? | Flexible string matching |
| `regexp` | Regular expression matching | Complex pattern matching |
| `fuzzy` | Tolerates spelling mistakes | Typo tolerance |
| `ngram` | Matches character n-grams | Partial word matching |
| `phrase` | Matches exact phrase in order | Exact phrase search |
| `span` | Positional queries | Near queries, ordered matching |
| `term` | Exact term matching (no analysis) | Exact value matching |
| `bool` | Complex boolean logic | AND/OR/NOT combinations |

**Example - Prefix Query for Autocomplete:**
```json
{
  "query": {
    "prefix": {
      "title": {
        "value": "open"
      }
    }
  }
}
```

**Example - Fuzzy Query for Typo Tolerance:**
```json
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "opensarch",
        "fuzziness": "AUTO"
      }
    }
  }
}
```

### 2.6 Language Support

| Feature | Support Level | Notes |
|---------|---------------|-------|
| English | ⭐⭐⭐⭐⭐ | Excellent with standard analyzer |
| Other Languages | ⭐⭐⭐⭐ | Requires language-specific analyzers |
| Cross-lingual | ⭐ | Not supported natively |
| CJK Languages | ⭐⭐⭐ | Requires specialized tokenizers (kuromoji, ik, etc.) |

### 2.7 When to Use BM25

✅ **Recommended:**
- Exact keyword/phrase search requirements
- Autocomplete and typeahead features
- Domain-specific terminology search
- Regex or wildcard pattern matching
- Maximum cost efficiency required
- Low-latency requirements at any scale

❌ **Not Recommended:**
- Semantic similarity search
- Cross-lingual search
- Synonym handling without manual configuration
- User queries differ significantly from document terminology

---

## 3. Dense Vector Search

### 3.1 Overview

Dense vector search uses neural network embeddings to represent text as dense floating-point vectors (typically 384-1536 dimensions). Similarity is computed using cosine similarity, dot product, or L2 distance.

### 3.2 Accuracy Characteristics

| Aspect | Rating | Notes |
|--------|--------|-------|
| Semantic Understanding | ⭐⭐⭐⭐⭐ | Captures meaning beyond keywords |
| Synonym Handling | ⭐⭐⭐⭐⭐ | Automatically handles synonyms |
| Cross-lingual | ⭐⭐⭐⭐⭐ | With multilingual models |
| Exact Match | ⭐⭐⭐ | May miss exact keyword matches |
| Domain-specific | ⭐⭐⭐ | Requires fine-tuning for specialized domains |

**Strengths:**
- Understands semantic meaning
- Handles paraphrases and synonyms naturally
- Supports cross-lingual search with multilingual models
- Zero-shot transfer to new domains

**Weaknesses:**
- May miss exact keyword matches
- Requires embedding model
- Higher computational cost
- Quality depends heavily on embedding model choice

### 3.3 Index Types Comparison

#### 3.3.1 HNSW (Hierarchical Navigable Small World)

**Overview:** Graph-based approximate nearest neighbor (ANN) algorithm. Default and most popular choice.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 95-99%+ recall achievable with proper tuning |
| **Build Time** | Moderate to slow |
| **Query Latency** | Fast (1-50ms typically) |
| **Memory Requirement** | High - entire graph in memory |
| **Scalability** | Good, but memory-bound |

**Parameters:**
- `m` (default: 16): Number of bi-directional links per node. Higher = more accurate but more memory
- `ef_construction` (default: 100): Build-time accuracy. Higher = better quality but slower build
- `ef_search` (default: 100): Query-time accuracy. Higher = better recall but slower query

**Memory Estimation:**
```
Memory = num_vectors × (dimensions × 4 bytes + m × 8 bytes + overhead)
Example: 10M vectors × 768 dims, m=16
Memory ≈ 10M × (768 × 4 + 16 × 8) ≈ 32GB
```

**Cost Profile:**
| Resource | Cost Level |
|----------|------------|
| Storage | ⭐⭐⭐ (Medium-High) |
| Memory | ⭐⭐⭐⭐⭐ (Very High) |
| CPU (Build) | ⭐⭐⭐ (Medium) |
| CPU (Query) | ⭐⭐ (Low-Medium) |

**Best For:**
- Small to medium datasets that fit in memory
- Low-latency requirements
- High accuracy requirements

#### 3.3.2 IVF (Inverted File Index)

**Overview:** Clustering-based approach that partitions vectors into clusters.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 85-95% recall typical |
| **Build Time** | Slow (requires training) |
| **Query Latency** | Medium (5-100ms) |
| **Memory Requirement** | Lower than HNSW |
| **Scalability** | Better for large datasets |

**Parameters:**
- `nlist`: Number of clusters. Typically sqrt(n) to n/1000
- `nprobe`: Clusters to search at query time. Higher = better recall, slower

**Memory Estimation:**
```
Memory = num_vectors × dimensions × 4 bytes + cluster_centroids
Much lower than HNSW as no graph structure
```

**Best For:**
- Larger datasets where memory is constrained
- Can tolerate slightly lower accuracy
- Batch search workloads

#### 3.3.3 DiskANN

**Overview:** Microsoft's disk-based ANN algorithm designed for billion-scale vector search with limited memory.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 95%+ recall achievable |
| **Build Time** | Slow |
| **Query Latency** | Medium (10-100ms), depends on SSD speed |
| **Memory Requirement** | Very Low - most data on disk |
| **Scalability** | Excellent for very large datasets |

**OpenSearch Implementation:**
- Enabled via `mode: disk` in k-NN settings
- Combines with Vamana graph algorithm
- Uses compressed vectors in memory, full vectors on disk

**Memory Estimation:**
```
Memory ≈ num_vectors × compressed_dims × bytes_per_dim
With 4x compression: 10M × 768 / 4 × 4 bytes ≈ 7.5GB
```

**Cost Profile:**
| Resource | Cost Level |
|----------|------------|
| Storage | ⭐⭐⭐⭐ (High - raw vectors + index) |
| Memory | ⭐⭐ (Low) |
| CPU (Query) | ⭐⭐⭐ (Medium) |
| SSD IOPS | ⭐⭐⭐⭐ (High - fast NVMe recommended) |

**Best For:**
- Billion-scale datasets
- Memory-constrained environments
- Cost-sensitive deployments with large data

#### 3.3.4 Product Quantization (PQ)

**Overview:** Compression technique that quantizes vectors into compact codes.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 80-90% recall (some accuracy loss) |
| **Memory Reduction** | 10-50x compression possible |
| **Query Latency** | Can be faster due to cache efficiency |
| **Build Complexity** | Requires codebook training |

**Parameters:**
- `m`: Number of sub-vectors (must divide dimensions evenly)
- `code_size`: Bits per sub-vector (typically 8)

**Memory Estimation:**
```
Compressed Memory = num_vectors × m × code_size / 8
Example: 10M vectors, m=64, code_size=8
Memory ≈ 10M × 64 × 1 byte = 640MB (vs 30GB uncompressed)
```

**Best For:**
- Memory is primary constraint
- Can tolerate accuracy loss
- Very large datasets

#### 3.3.5 Binary Quantization (BQ)

**Overview:** Extreme compression using binary vectors.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 70-85% recall (significant accuracy loss) |
| **Memory Reduction** | 32x compression |
| **Query Latency** | Very fast (uses hardware bit operations) |
| **Build Time** | Fast |

**Memory Estimation:**
```
Binary Memory = num_vectors × dimensions / 8
Example: 10M × 768 dims → 10M × 96 bytes = 960MB
```

**Best For:**
- First-stage candidate retrieval
- Extremely large datasets
- Must be combined with re-ranking for accuracy

### 3.4 Dense Vector Latency Scaling

| Data Scale | HNSW | IVF | DiskANN | PQ |
|------------|------|-----|---------|-----|
| 100K | 1-5ms | 5-15ms | 10-30ms | 2-8ms |
| 1M | 2-10ms | 10-30ms | 15-50ms | 5-15ms |
| 10M | 5-20ms | 20-60ms | 20-80ms | 10-30ms |
| 100M | 10-50ms | 50-150ms | 30-100ms | 20-60ms |
| 1B | Memory limit | 100-300ms | 50-200ms | 50-150ms |

*Note: Latencies are approximate and depend heavily on hardware, configuration, and query patterns.*

### 3.5 Embedding Dimensions Trade-offs

| Dimensions | Accuracy | Storage/Memory | Latency | Example Models |
|------------|----------|----------------|---------|----------------|
| 384 | Good | Low | Fast | all-MiniLM-L6-v2, paraphrase-MiniLM |
| 768 | Very Good | Medium | Medium | all-mpnet-base-v2, BERT-base |
| 1024 | Excellent | High | Slower | BGE-large, E5-large |
| 1536 | Excellent | Very High | Slowest | OpenAI text-embedding-3-large, Cohere |

### 3.6 Language Support

| Feature | Support Level | Notes |
|---------|---------------|-------|
| English | ⭐⭐⭐⭐⭐ | Excellent with most models |
| Multilingual | ⭐⭐⭐⭐⭐ | With multilingual models (mE5, multilingual-e5, etc.) |
| Cross-lingual | ⭐⭐⭐⭐⭐ | Query in one language, retrieve in another |
| Low-resource Languages | ⭐⭐⭐ | Depends on model training data |

### 3.7 When to Use Dense Vector

✅ **Recommended:**
- Semantic similarity search
- Cross-lingual search requirements
- Synonym and paraphrase handling needed
- Natural language queries from users
- Question-answering systems
- RAG (Retrieval Augmented Generation) applications

❌ **Not Recommended:**
- Exact keyword matching is critical
- Highly specialized domain vocabulary not covered by model
- Extremely cost-sensitive deployments
- Real-time autocomplete/typeahead
- Sub-millisecond latency requirements

---

## 4. Sparse Vector Search

### 4.1 Overview

Sparse vector search uses learned sparse representations where most dimensions are zero. Unlike dense vectors with 384-1536 dimensions all populated, sparse vectors may have 30,000+ dimensions but only 100-500 non-zero values.

### 4.2 Types of Sparse Vector Implementations

#### 4.2.1 Neural Sparse (SPLADE, etc.)

**Overview:** Uses neural networks to learn sparse representations with semantic meaning.

**How it works:**
1. Documents and queries are encoded into sparse vectors
2. Each dimension corresponds to a vocabulary token
3. Weights indicate semantic importance (not just term frequency)

**Advantages over BM25:**
- Learns semantic term expansion (e.g., "dog" activates "puppy", "canine")
- Trained on relevance signals
- Better zero-shot domain transfer

**OpenSearch Neural Sparse Pipeline:**
```json
{
  "description": "Neural sparse ingestion pipeline",
  "processors": [
    {
      "sparse_encoding": {
        "model_id": "model_id_here",
        "field_map": {
          "passage_text": "passage_embedding"
        }
      }
    }
  ]
}
```

#### 4.2.2 Index Backends for Sparse Vectors

##### A. rank_features Field (Inverted Index Based)

**Overview:** Uses OpenSearch's native inverted index structure optimized for sparse features.

| Aspect | Details |
|--------|---------|
| **Accuracy** | Exact (no approximation) |
| **Query Latency** | Scales with vocabulary overlap |
| **Memory** | Moderate |
| **Index Size** | Similar to text fields |

**Mapping:**
```json
{
  "mappings": {
    "properties": {
      "sparse_embedding": {
        "type": "rank_features"
      }
    }
  }
}
```

**Query:**
```json
{
  "query": {
    "neural_sparse": {
      "sparse_embedding": {
        "query_text": "semantic search opensearch",
        "model_id": "model_id_here"
      }
    }
  }
}
```

**Best For:**
- Exact sparse vector search
- Smaller datasets (< 50M documents)
- When accuracy is paramount

##### B. SEISMIC (ANN-based Sparse Search)

**Overview:** Approximate nearest neighbor algorithm specifically designed for sparse vectors. Available in OpenSearch 2.17+.

| Aspect | Details |
|--------|---------|
| **Accuracy** | 95%+ recall achievable |
| **Query Latency** | Much faster than rank_features for large data |
| **Memory** | Moderate |
| **Build Time** | Slower than rank_features |

**When to Use SEISMIC:**
- Large-scale datasets (> 10M documents)
- Latency-sensitive applications
- Can tolerate slight approximation

**Latency Comparison:**
| Data Scale | rank_features | SEISMIC |
|------------|---------------|---------|
| 1M docs | 10-50ms | 5-20ms |
| 10M docs | 50-200ms | 10-40ms |
| 100M docs | 200-800ms | 20-80ms |

### 4.3 Accuracy Characteristics

| Aspect | Rating | Notes |
|--------|--------|-------|
| Semantic Understanding | ⭐⭐⭐⭐ | Good, but generally slightly below dense |
| Exact Match | ⭐⭐⭐⭐ | Better than dense vectors |
| Term Expansion | ⭐⭐⭐⭐⭐ | Learns relevant term expansion |
| Interpretability | ⭐⭐⭐⭐⭐ | Can see which terms matched |

### 4.4 Cost Profile

| Resource | Cost Level | Notes |
|----------|------------|-------|
| Storage | ⭐⭐ (Low-Medium) | Only non-zero values stored |
| Memory | ⭐⭐ (Low-Medium) | Efficient inverted index structure |
| CPU (Indexing) | ⭐⭐⭐ (Medium) | Requires model inference |
| CPU (Query) | ⭐⭐ (Low-Medium) | Efficient inverted index lookup |

**Storage Estimation:**
```
Storage ≈ num_docs × avg_non_zero_terms × (term_id_bytes + weight_bytes)
Example: 10M docs × 200 terms × 8 bytes ≈ 16GB
```

### 4.5 Latency Characteristics

| Data Scale | rank_features | SEISMIC |
|------------|---------------|---------|
| < 1M docs | 5-30ms | 3-15ms |
| 1M-10M docs | 20-100ms | 10-40ms |
| 10M-100M docs | 100-500ms | 30-100ms |
| > 100M docs | 300-1000ms | 50-200ms |

### 4.6 Sparse Encoding Models

| Model | Vocabulary Size | Avg Non-zero Terms | Quality | Speed |
|-------|-----------------|-------------------|---------|-------|
| SPLADE++ | 30,522 (BERT) | 100-300 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| SPLADE-v3 | 30,522 | 50-150 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| neural-sparse-encoding-v1 | 30,522 | 100-200 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| opensearch-neural-sparse-* | 30,522 | 100-200 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4.7 Two-Phase Sparse Search (Doc-only Mode)

**Overview:** Optimization where only documents are encoded with neural sparse model, and queries use simple tokenization.

**Benefits:**
- 10x+ faster query latency (no model inference at query time)
- Good for high-throughput scenarios
- Slight accuracy trade-off

**Accuracy Comparison:**
| Mode | Latency | Accuracy |
|------|---------|----------|
| Bi-encoder (both) | Baseline | Baseline |
| Doc-only | 10x faster | 5-10% lower |

### 4.8 Language Support

| Feature | Support Level | Notes |
|---------|---------------|-------|
| English | ⭐⭐⭐⭐⭐ | Excellent |
| Other Languages | ⭐⭐⭐ | Model-dependent |
| Cross-lingual | ⭐⭐ | Limited, less effective than dense |

### 4.9 When to Use Sparse Vector

✅ **Recommended:**
- Balance between lexical and semantic search
- Interpretability is important (can see which terms matched)
- Domain with specialized vocabulary
- Hybrid with BM25 (complementary signals)
- Lower memory budget than dense vectors

❌ **Not Recommended:**
- Cross-lingual search (dense vectors better)
- Maximum semantic understanding needed
- Very short queries (dense often better)

---

## 5. Hybrid Search

### 5.1 Overview

Hybrid search combines multiple retrieval methods (BM25, dense vector, sparse vector) to leverage the strengths of each. OpenSearch supports hybrid search through the hybrid query type and score normalization.

### 5.2 Score Normalization Methods

#### 5.2.1 Min-Max Normalization
```
normalized_score = (score - min_score) / (max_score - min_score)
```
- Simple and intuitive
- Can be skewed by outliers

#### 5.2.2 L2 Normalization
```
normalized_score = score / sqrt(sum(scores²))
```
- Less sensitive to outliers
- Maintains relative differences

#### 5.2.3 Arithmetic Mean Combination
```
final_score = (score_1 + score_2) / 2
```

#### 5.2.4 Harmonic Mean Combination
```
final_score = 2 / (1/score_1 + 1/score_2)
```
- Penalizes cases where one score is very low

#### 5.2.5 Geometric Mean Combination
```
final_score = sqrt(score_1 × score_2)
```

### 5.3 OpenSearch Hybrid Query Syntax

```json
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "text_field": "search query"
          }
        },
        {
          "neural": {
            "vector_field": {
              "query_text": "search query",
              "model_id": "model_id",
              "k": 100
            }
          }
        }
      ]
    }
  },
  "search_pipeline": "hybrid_search_pipeline"
}
```

**Search Pipeline Configuration:**
```json
{
  "description": "Hybrid search pipeline",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.3, 0.7]
          }
        }
      }
    }
  ]
}
```

### 5.4 Hybrid Combinations

| Combination | Strengths | Best For |
|-------------|-----------|----------|
| BM25 + Dense | Exact match + semantic | General purpose semantic search |
| BM25 + Sparse | Lexical + learned expansion | Domain-specific search |
| Dense + Sparse | Dual semantic signals | Maximum semantic coverage |
| BM25 + Dense + Sparse | All three methods | Highest accuracy possible |

### 5.5 Accuracy Improvement

Typical accuracy gains from hybrid search:

| Base Method | + BM25 | + Dense | + Sparse |
|-------------|--------|---------|----------|
| BM25 | - | +5-15% | +3-10% |
| Dense | +5-15% | - | +2-8% |
| Sparse | +3-10% | +2-8% | - |

*Note: Gains vary significantly based on dataset and queries.*

### 5.6 Cost Profile

| Component | Additional Cost |
|-----------|-----------------|
| Storage | Sum of all method storage |
| Memory | Sum of all method memory |
| CPU (Indexing) | Sum of all indexing costs |
| CPU (Query) | Sum of query costs + normalization overhead |

**Rule of Thumb:**
- Hybrid of 2 methods ≈ 1.8-2x cost of single method
- Hybrid of 3 methods ≈ 2.5-3x cost of single method

### 5.7 Latency Characteristics

Hybrid query latency is typically determined by the slowest component plus combination overhead.

| Components | Typical Latency Overhead |
|------------|-------------------------|
| 2 methods in parallel | +10-20% over slowest |
| 3 methods in parallel | +15-30% over slowest |
| Sequential execution | Sum of all latencies |

OpenSearch executes hybrid sub-queries in parallel when possible.

### 5.8 Weight Tuning Guidelines

| Scenario | BM25 Weight | Dense Weight | Sparse Weight |
|----------|-------------|--------------|---------------|
| General web search | 0.3 | 0.7 | - |
| Technical documentation | 0.5 | 0.5 | - |
| E-commerce (product names) | 0.6 | 0.4 | - |
| Semantic Q&A | 0.2 | 0.8 | - |
| Domain-specific + semantic | 0.3 | 0.4 | 0.3 |

### 5.9 When to Use Hybrid Search

✅ **Recommended:**
- Maximum recall and precision needed
- Mixed query types (some exact, some semantic)
- Unknown query distribution
- Can afford additional infrastructure cost
- Production search applications

❌ **Not Recommended:**
- Strict cost constraints
- Simple use cases where one method suffices
- Sub-10ms latency requirements
- Development/prototype phase (start simple)

---

## 6. Model Deployment Options

### 6.1 Overview of Deployment Options

| Option | Latency | Cost | Scalability | Maintenance |
|--------|---------|------|-------------|-------------|
| External API | 20-200ms | Per-request | Infinite | None |
| OpenSearch ML Node (CPU) | 10-100ms | Fixed | Manual scaling | Medium |
| OpenSearch ML Node (GPU) | 5-30ms | Higher fixed | Manual scaling | Medium |
| Remote SageMaker | 10-50ms | Per-request or endpoint | Auto-scaling | Low |
| Remote Custom Endpoint | Variable | Variable | Custom | High |

### 6.2 External API Services

#### Available Services:

| Provider | Models | Latency | Cost |
|----------|--------|---------|------|
| OpenAI | text-embedding-3-small/large | 50-200ms | $0.02-0.13 per 1M tokens |
| Cohere | embed-english-v3, embed-multilingual | 50-150ms | $0.10 per 1M tokens |
| Amazon Bedrock | Titan Embeddings, Cohere | 30-100ms | $0.02-0.10 per 1M tokens |
| Azure OpenAI | text-embedding-ada-002, etc. | 50-200ms | Similar to OpenAI |
| Google Vertex AI | textembedding-gecko | 50-150ms | $0.025 per 1M characters |

**Pros:**
- No infrastructure management
- Access to state-of-the-art models
- Automatic updates and improvements
- Pay-per-use pricing

**Cons:**
- Network latency added to every request
- Cost can be high at scale
- Data leaves your network
- Rate limits may apply
- Vendor lock-in risk

**OpenSearch Connector Configuration (Example for OpenAI):**
```json
{
  "name": "openai-embedding-connector",
  "description": "OpenAI text-embedding-3-small",
  "version": 1,
  "protocol": "http",
  "parameters": {
    "model": "text-embedding-3-small"
  },
  "credential": {
    "openai_api_key": "your-api-key"
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://api.openai.com/v1/embeddings",
      "headers": {
        "Authorization": "Bearer ${credential.openai_api_key}"
      },
      "request_body": "{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
      "pre_process_function": "connector.pre_process.openai.embedding",
      "post_process_function": "connector.post_process.openai.embedding"
    }
  ]
}
```

**Cost Estimation:**
```
Monthly Cost = (docs × avg_tokens + queries × avg_query_tokens) × price_per_1M_tokens / 1,000,000

Example: 10M docs (avg 500 tokens) + 1M queries (avg 20 tokens) with OpenAI small:
= (10M × 500 + 1M × 20) × $0.02 / 1M
= (5B + 20M) × $0.02 / 1M
= ~$100,400 for initial indexing + $0.40/month for queries
```

### 6.3 OpenSearch ML Node - CPU Deployment

**Overview:** Deploy models directly on OpenSearch ML nodes using CPU inference.

**Supported Model Types:**
- Sentence Transformers models (via PyTorch/ONNX)
- OpenSearch pre-trained models
- Custom models (with proper format)

**Configuration:**
```yaml
# opensearch.yml
plugins:
  ml_commons:
    only_run_on_ml_node: true
    model_access_control_enabled: true
    native_memory_threshold: 90
```

**Resource Requirements:**

| Model Size | Memory Required | Recommended Instance |
|------------|-----------------|---------------------|
| Small (< 100M params) | 2-4GB | 8GB RAM, 4 vCPU |
| Medium (100-300M params) | 4-8GB | 16GB RAM, 8 vCPU |
| Large (> 300M params) | 8-16GB | 32GB RAM, 16 vCPU |

**Latency Characteristics:**

| Model Size | Latency (single) | Throughput |
|------------|------------------|------------|
| MiniLM (22M) | 5-15ms | 100-200 req/s per node |
| MPNet (110M) | 20-50ms | 30-60 req/s per node |
| Large (335M) | 50-150ms | 10-25 req/s per node |

**Pros:**
- No external dependencies
- Data stays within cluster
- Predictable costs
- Lower latency than external APIs

**Cons:**
- Must manage model lifecycle
- Limited to models that fit in memory
- CPU inference slower than GPU
- Scaling requires cluster changes

**Deploy Model Example:**
```json
// Register model
POST /_plugins/_ml/models/_register
{
  "name": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "1.0.1",
  "model_format": "TORCH_SCRIPT"
}

// Deploy model
POST /_plugins/_ml/models/{model_id}/_deploy
```

### 6.4 OpenSearch ML Node - GPU Deployment

**Overview:** Use GPU-equipped ML nodes for faster inference.

**Benefits over CPU:**
- 5-10x faster inference
- Higher throughput
- Better for larger models

**Hardware Options:**

| GPU | Memory | Typical Latency | Cost/hour (AWS) |
|-----|--------|-----------------|-----------------|
| NVIDIA T4 | 16GB | 5-15ms | $0.50-1.00 |
| NVIDIA A10G | 24GB | 3-10ms | $1.00-1.50 |
| NVIDIA A100 | 40/80GB | 2-8ms | $3.00-5.00 |

**Pros:**
- Fastest on-premise inference
- Handles larger models
- High throughput

**Cons:**
- Higher infrastructure cost
- Requires GPU node management
- Limited GPU availability in some regions

### 6.5 Amazon SageMaker Integration

**Overview:** Deploy models on SageMaker endpoints and connect to OpenSearch.

**Deployment Options:**

| Type | Use Case | Latency | Cost Model |
|------|----------|---------|------------|
| Real-time Endpoint | Low-latency inference | 10-50ms | Per-hour + per-request |
| Serverless Inference | Variable traffic | 50-200ms (cold start) | Per-request only |
| Async Inference | Batch processing | Minutes | Per-request |

**SageMaker Connector Example:**
```json
{
  "name": "sagemaker-embedding-connector",
  "description": "SageMaker embedding endpoint",
  "version": 1,
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-east-1",
    "service_name": "sagemaker"
  },
  "credential": {
    "access_key": "...",
    "secret_key": "...",
    "session_token": "..."
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-embedding-endpoint/invocations",
      "headers": {
        "Content-Type": "application/json"
      },
      "request_body": "{ \"inputs\": ${parameters.input} }"
    }
  ]
}
```

**Cost Estimation (Real-time Endpoint):**
```
Monthly Cost = instance_cost × hours × instances + request_cost × requests

Example: ml.g4dn.xlarge ($0.7364/hr), 24/7, 2 instances
= $0.7364 × 24 × 30 × 2 = ~$1,060/month
+ per-request charges (usually minimal)
```

**Pros:**
- Managed infrastructure
- Easy auto-scaling
- GPU instances available
- Pay-per-use options

**Cons:**
- Network latency to SageMaker
- Additional AWS service complexity
- Can be expensive for high-volume

### 6.6 Custom Model Deployment

**For deploying custom/fine-tuned models:**

**Options:**
1. **Convert to OpenSearch format:**
   - Export to TorchScript or ONNX
   - Register with ML Commons

2. **Deploy to SageMaker:**
   - Create custom inference container
   - Deploy as SageMaker endpoint

3. **Self-hosted inference server:**
   - Use TorchServe, Triton, or vLLM
   - Connect via HTTP connector

**Model Conversion Example (PyTorch to TorchScript):**
```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('your-fine-tuned-model')
dummy_input = model.tokenize(['example text'])
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('model.pt')
```

### 6.7 Deployment Decision Matrix

| Requirement | Recommended Option |
|-------------|-------------------|
| Lowest operational overhead | External API (OpenAI, Cohere) |
| Lowest query latency | GPU ML Node or SageMaker GPU |
| Lowest cost at scale | CPU ML Node (amortized) |
| Data sovereignty | CPU/GPU ML Node (on-prem) |
| Variable traffic | SageMaker Serverless or API |
| Custom model required | SageMaker or ML Node |
| Fastest time-to-production | External API |
| Maximum control | Self-hosted inference server |

---

## 7. Decision Matrix

### 7.1 By Data Scale

| Data Scale | Recommended Primary | Index Type | Deployment |
|------------|---------------------|------------|------------|
| < 100K docs | Any | HNSW | CPU ML Node or API |
| 100K - 1M | Dense or Hybrid | HNSW | CPU ML Node |
| 1M - 10M | Hybrid | HNSW or IVF | CPU/GPU ML Node |
| 10M - 100M | Hybrid | DiskANN or IVF | SageMaker or GPU |
| > 100M | Hybrid | DiskANN + PQ | Distributed SageMaker |

### 7.2 By Latency Requirements

| Latency Requirement | Recommended Approach |
|--------------------|----------------------|
| < 10ms | BM25 only, or small HNSW with GPU |
| 10-50ms | HNSW with CPU ML Node, small models |
| 50-100ms | Hybrid (BM25 + Dense), standard models |
| 100-500ms | Full hybrid, external APIs acceptable |
| > 500ms | Batch processing, async inference |

### 7.3 By Cost Budget

| Budget Level | Recommended Approach |
|--------------|----------------------|
| Minimal | BM25 only |
| Low | BM25 + Doc-only Sparse |
| Medium | Dense (small model, HNSW) + BM25 |
| High | Full Hybrid (BM25 + Dense + Sparse) |
| Unlimited | Full Hybrid with GPU, best models |

### 7.4 By Language Requirements

| Requirement | Recommended Approach |
|-------------|----------------------|
| English only | Any method works well |
| Single non-English | Dense with language-specific model |
| Multilingual (same lang query/doc) | Multilingual dense model |
| Cross-lingual (query ≠ doc language) | Multilingual dense (mE5, multilingual-e5) |
| CJK languages | Dense + BM25 with proper tokenizer |

### 7.5 By Accuracy Requirements

| Accuracy Level | Recommended Approach |
|----------------|----------------------|
| Basic (exploratory) | BM25 or single method |
| Standard | Dense vector search |
| High | Hybrid (BM25 + Dense) |
| Maximum | Triple Hybrid + Re-ranking |

---

## 8. Quick Reference Tables

### 8.1 Method Comparison Summary

| Aspect | BM25 | Dense Vector | Sparse Vector | Hybrid |
|--------|------|--------------|---------------|--------|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Semantic** | ❌ | ✅ | ✅ | ✅ |
| **Exact Match** | ✅ | ⚠️ | ✅ | ✅ |
| **Storage Cost** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Cost** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Query Latency** | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Cross-lingual** | ❌ | ✅ | ⚠️ | ✅ |
| **Setup Complexity** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 8.2 Index Type Comparison

| Index Type | Memory | Query Speed | Accuracy | Scale Limit |
|------------|--------|-------------|----------|-------------|
| HNSW | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~100M vectors |
| IVF | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~500M vectors |
| DiskANN | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Billions |
| PQ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~1B vectors |
| BQ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~1B vectors |
| rank_features | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~50M docs |
| SEISMIC | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~500M docs |

### 8.3 Model Deployment Comparison

| Option | Setup Time | Latency | Cost (Low Vol) | Cost (High Vol) | Data Privacy |
|--------|------------|---------|----------------|-----------------|--------------|
| OpenAI API | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| Cohere API | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| Bedrock | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| ML Node CPU | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| ML Node GPU | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| SageMaker | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |

*Lower stars = better (less time, lower latency, lower cost)*

### 8.4 Common Configurations

#### Configuration A: Cost-Optimized Semantic Search
```
Method: BM25 + Dense (doc-only encoding)
Index: HNSW (m=8, ef=100)
Model: MiniLM (384 dims) on CPU ML Node
Est. Cost: $0.10 / 1M queries
```

#### Configuration B: Balanced Production
```
Method: Hybrid (BM25 0.3 + Dense 0.7)
Index: HNSW (m=16, ef=256)
Model: MPNet (768 dims) on CPU ML Node
Est. Cost: $0.50 / 1M queries
```

#### Configuration C: Maximum Accuracy
```
Method: Triple Hybrid (BM25 + Dense + Sparse)
Index: HNSW (m=32, ef=512)
Model: Large model (1024 dims) on GPU + Sparse model
Est. Cost: $2-5 / 1M queries
```

#### Configuration D: Large Scale (100M+ docs)
```
Method: Hybrid (BM25 + Dense)
Index: DiskANN with PQ compression
Model: API service (batched) or SageMaker
Est. Cost: Variable, depends on query volume
```

---

## Appendix A: OpenSearch Version Feature Matrix

| Feature | 2.9 | 2.11 | 2.13 | 2.15 | 2.17+ |
|---------|-----|------|------|------|-------|
| Neural Search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hybrid Query | ✅ | ✅ | ✅ | ✅ | ✅ |
| Neural Sparse | ❌ | ✅ | ✅ | ✅ | ✅ |
| DiskANN | ❌ | ❌ | ✅ | ✅ | ✅ |
| SEISMIC | ❌ | ❌ | ❌ | ❌ | ✅ |
| Binary Quantization | ❌ | ❌ | ❌ | ✅ | ✅ |
| Remote Model Connectors | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor - algorithms that find approximate (not exact) nearest neighbors for speed |
| **BM25** | Best Matching 25 - probabilistic ranking function for lexical search |
| **Dense Vector** | Fixed-length floating-point vector where all dimensions have values |
| **Sparse Vector** | Vector where most dimensions are zero; only non-zero values stored |
| **HNSW** | Hierarchical Navigable Small World - graph-based ANN algorithm |
| **IVF** | Inverted File Index - clustering-based ANN algorithm |
| **DiskANN** | Disk-based ANN algorithm for large-scale search |
| **PQ** | Product Quantization - vector compression technique |
| **BQ** | Binary Quantization - extreme vector compression to binary |
| **SPLADE** | Sparse Lexical and Expansion - neural sparse encoding method |
| **Hybrid Search** | Combining multiple search methods with score fusion |
| **ML Node** | OpenSearch node dedicated to machine learning workloads |

---

*Document Version: 1.0*
*Last Updated: January 2025*
*Applicable OpenSearch Versions: 2.9+*

