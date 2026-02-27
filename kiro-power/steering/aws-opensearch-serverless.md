---
title: "AWS OpenSearch Serverless Deployment"
inclusion: manual
---

# AWS OpenSearch Serverless Deployment Guide

This guide covers deploying your local OpenSearch search strategy to AWS OpenSearch Serverless.

## When to Use OpenSearch Serverless

Use OpenSearch Serverless for:
- Semantic search applications
- Hybrid search (BM25 + vector)
- Standard vector search workloads
- Applications requiring automatic scaling
- Cost-sensitive workloads with variable traffic
- Quick proof-of-concept deployments

**Do NOT use for Agentic Search** - use OpenSearch Domain instead (see aws-opensearch-domain.md).

## Prerequisites

Before starting Phase 5 deployment:
1. AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
2. Appropriate IAM permissions for OpenSearch Serverless
3. Successful Phase 4 execution with local OpenSearch running
4. Search strategy manifest file created in Phase 3

## Deployment Steps

### Step 1: Create OpenSearch Serverless Collection

Use the AWS API MCP server to create a serverless collection:

```json
POST /opensearchserverless/CreateCollection
{
  "name": "<collection-name>",
  "type": "SEARCH",
  "description": "Search application deployed from local OpenSearch"
}
```

Choose collection type based on the search strategy:
- **VECTORSEARCH**: For dense vector search workloads (semantic search with dense embeddings)
- **SEARCH**: For all other search workloads (BM25, neural sparse, hybrid with neural sparse)

Note: Neural sparse (automatic semantic enrichment) requires SEARCH type, not VECTORSEARCH.

### Step 2: Configure Network Access

Create a network policy for the collection using AWS API MCP:

```json
POST /opensearchserverless/CreateAccessPolicy
{
  "name": "<collection-name>-network-policy",
  "type": "network",
  "policy": "[{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/<collection-name>\"],\"AllowFromPublic\":true}],\"AllowFromPublic\":true}]"
}
```

Network policy options:
- **Public access (for development)**: Set `AllowFromPublic: true`
- **VPC endpoint access (for production)**: Specify VPC endpoint IDs in the policy

### Step 3: Configure Data Access

Create a data access policy with appropriate permissions using AWS API MCP:

```json
POST /opensearchserverless/CreateAccessPolicy
{
  "name": "<collection-name>-data-policy",
  "type": "data",
  "policy": "[{\"Rules\":[{\"ResourceType\":\"index\",\"Resource\":[\"index/<collection-name>/*\"],\"Permission\":[\"aoss:CreateIndex\",\"aoss:DescribeIndex\",\"aoss:UpdateIndex\",\"aoss:DeleteIndex\",\"aoss:ReadDocument\",\"aoss:WriteDocument\"]},{\"ResourceType\":\"collection\",\"Resource\":[\"collection/<collection-name>\"],\"Permission\":[\"aoss:CreateCollectionItems\",\"aoss:DescribeCollectionItems\"]},{\"ResourceType\":\"model\",\"Resource\":[\"model/<collection-name>/*\"],\"Permission\":[\"aoss:CreateMLResource\"]}],\"Principal\":[\"arn:aws:iam::<account-id>:role/<role-name>\"]}]"
}
```

This policy grants permissions for:
- **Index**: Create, update, describe, delete indices and read/write documents
- **Collection**: Create and describe collection items (required for pipelines)
- **Model**: Create ML resources (required for automatic semantic enrichment)

Replace `<account-id>` and `<role-name>` with the appropriate AWS principal.

**For private collections**, also configure network access to allow `aoss.amazonaws.com` service access.

### Step 4: Wait for Collection to be Active

Poll collection status until active using AWS API MCP:

```json
POST /opensearchserverless/BatchGetCollection
{
  "names": ["<collection-name>"]
}
```

Wait for status: "ACTIVE" (typically takes 1-3 minutes)

### Step 5: Create Index with Automatic Semantic Enrichment (Neural Sparse)

**For Neural Sparse search strategies**, use automatic semantic enrichment:

OpenSearch Serverless supports automatic semantic enrichment for Neural Sparse, which automatically manages models and pipelines. Use the AWS API MCP to create the index:

```json
POST /opensearchserverless/CreateIndex
{
  "id": "<collection-id>",
  "indexName": "<index-name>",
  "indexSchema": {
    "mappings": {
      "properties": {
        "<text-field>": {
          "type": "text",
          "semantic_enrichment": {
            "status": "ENABLED",
            "language_options": "english"
          }
        }
      }
    }
  }
}
```

Key points about automatic semantic enrichment:
- Set `semantic_enrichment.status` to "ENABLED" on text fields that should use neural sparse
- Specify `language_options`: "english" or "multi-lingual" (supports 15 languages including Arabic, Bengali, Chinese, Finnish, French, Hindi, Indonesian, Japanese, Korean, Persian, Russian, Spanish, Swahili, Telugu)
- You can have both semantic and non-semantic text fields in the same index
- The system automatically:
  - Deploys the service-managed sparse model
  - Creates ingest pipelines for document enrichment
  - Creates search pipelines for query enrichment
  - Rewrites "match" queries to neural sparse queries (no query changes needed)
- No manual model or pipeline management required
- Best for small-to-medium sized fields with natural language content (product descriptions, reviews, summaries)
- Token limits: 8,192 tokens for English, 512 tokens for multilingual
- Improves relevance by ~20% for English, ~105% for multilingual over BM25
- Charged based on OCU consumption during indexing only (monitor with SemanticSearchOCU CloudWatch metric)

**For other search strategies** (BM25, dense vector, hybrid with dense vectors):

Use the opensearch-mcp-server tools to create the index on the collection endpoint:

1. Get the local index configuration from the manifest
2. Create the index on the serverless collection endpoint
3. Include all mappings, settings, and configurations from local setup

### Step 6: Deploy ML Models (if applicable)

**For Neural Sparse**: Skip this step - automatic semantic enrichment handles everything.

**For Dense Vector embeddings** (semantic/hybrid search):

OpenSearch Serverless supports:
1. **Amazon Bedrock integration** (recommended):
   - No model deployment needed
   - Use Bedrock connector in pipelines
   - Supports models like Titan Embeddings, Cohere Embed
   
2. **Remote models via connectors**:
   - SageMaker endpoints
   - External API endpoints

Update pipeline configurations to use AWS-hosted models instead of local models.

### Step 7: Create Ingest Pipelines (if needed)

**For Neural Sparse with automatic semantic enrichment**: Skip this step - pipelines are automatically created and managed.

**For other search strategies**:

Recreate ingest pipelines on the serverless collection:

1. Get pipeline definitions from local setup
2. Create pipelines using opensearch-mcp-server
3. Update processor configurations for AWS (e.g., Bedrock connectors)
4. Attach pipelines to the index

### Step 8: Index Sample Documents

### Step 8: Index Sample Documents

Index test documents to verify the setup:

1. Use the same sample documents from Phase 1
2. For Neural Sparse with automatic enrichment:
   - Documents are automatically enriched during ingestion
   - Sparse vectors are generated and stored
   - No additional configuration needed
3. For other strategies: Verify embeddings are generated correctly
4. Test search queries to confirm functionality
5. For Neural Sparse: Use standard "match" queries - they're automatically rewritten to neural sparse queries

Give the user:
- Collection endpoint URL
- Collection ARN
- Dashboard URL (if applicable)
- Sample search queries to test
- Cost estimation based on collection type and expected usage

## Cost Considerations

OpenSearch Serverless pricing:
- Charged for OCU (OpenSearch Compute Units) hours
- Minimum: 2 OCUs for indexing, 2 OCUs for search
- Scales automatically based on workload
- Storage charged separately per GB

Recommend monitoring costs in AWS Cost Explorer.

## Security Best Practices

1. Use IAM roles instead of access keys when possible
2. Enable encryption at rest (enabled by default)
3. Use VPC endpoints for production workloads
4. Implement least-privilege access policies
5. Enable CloudWatch logging for audit trails

## Troubleshooting

Common issues:
- **Access denied**: Check data access policy and IAM permissions
- **Collection creation fails**: Verify service quotas and region availability
- **Model deployment fails**: Ensure Bedrock models are available in the region
- **Search returns no results**: Verify index mappings and pipeline configurations

## Next Steps

After successful deployment:
1. Update application code to use the serverless endpoint
2. Set up monitoring and alerting in CloudWatch
3. Configure backup strategies if needed
4. Plan for production scaling and optimization
