# Requirements Document

## Introduction

This document specifies the requirements for converting the existing OpenSearch Solution Architect multi-agent system (built on the Strands framework) into a Kiro Power with MCP (Model Context Protocol) server integration. The current system uses an orchestrator agent that delegates to sub-agents (Solution Planning Assistant, QA Assistant, Worker Agent) through interactive loops. The target architecture replaces the nested agent orchestration with a flat set of 13 MCP tools organized into three categories: Knowledge Tools, Workflow Tools, and Operation Tools. Kiro handles orchestration directly, and each tool is stateless with all context passed as parameters.

## Glossary

- **MCP_Server**: The Model Context Protocol server that exposes tools to Kiro for invocation
- **Knowledge_Tool**: An MCP tool that reads static knowledge base content or searches live documentation
- **Workflow_Tool**: An MCP tool that performs expert analysis or recommendations using Strands agent calls internally
- **Operation_Tool**: An MCP tool that performs CRUD operations against an OpenSearch cluster
- **Kiro_Power**: A packaged extension for Kiro consisting of an MCP server, configuration, and documentation
- **OpenSearch_Client**: The Python client used to communicate with an OpenSearch cluster
- **Knowledge_Base**: The set of markdown guide files containing OpenSearch semantic search expertise
- **Strands_Agent**: An LLM-based agent built with the Strands framework, used internally by Workflow Tools
- **Tool_Parameter**: An explicit input argument passed to an MCP tool, replacing global state

## Requirements

### Requirement 1: MCP Server Entry Point

**User Story:** As a Kiro user, I want the Power to expose all tools through a single MCP server, so that Kiro can discover and invoke them seamlessly.

#### Acceptance Criteria

1. WHEN the MCP server starts, THE MCP_Server SHALL register all 13 tools (4 Knowledge Tools, 3 Workflow Tools, 6 Operation Tools) and make them available for invocation
2. WHEN Kiro sends a tool invocation request, THE MCP_Server SHALL route the request to the correct tool handler and return the result
3. IF the MCP server encounters a startup error, THEN THE MCP_Server SHALL log a descriptive error message and exit with a non-zero status code
4. THE MCP_Server SHALL use the stdio transport for communication with Kiro

### Requirement 2: Knowledge Base Reading Tools

**User Story:** As a Kiro user, I want to access OpenSearch knowledge base content through dedicated tools, so that I can get expert guidance on search strategies, dense vector models, and sparse vector models.

#### Acceptance Criteria

1. WHEN the `read_knowledge_base` tool is invoked, THE Knowledge_Tool SHALL return the full content of the OpenSearch Semantic Search Guide markdown file
2. WHEN the `read_dense_vector_models` tool is invoked, THE Knowledge_Tool SHALL return the full content of the Dense Vector Models Guide markdown file
3. WHEN the `read_sparse_vector_models` tool is invoked, THE Knowledge_Tool SHALL return the full content of the Sparse Vector Models Guide markdown file
4. IF a knowledge base file is missing or unreadable, THEN THE Knowledge_Tool SHALL return a descriptive error message indicating the file path and failure reason
5. THE Knowledge_Tool SHALL resolve file paths relative to the MCP server package directory, independent of the working directory

### Requirement 3: Live Documentation Search Tool

**User Story:** As a Kiro user, I want to search the opensearch.org documentation for the latest information, so that I can access up-to-date details beyond the static knowledge base.

#### Acceptance Criteria

1. WHEN the `search_opensearch_org` tool is invoked with a query string, THE Knowledge_Tool SHALL return search results from opensearch.org containing title, URL, and snippet for each result
2. WHEN the `search_opensearch_org` tool is invoked with a `numberOfResults` parameter, THE Knowledge_Tool SHALL return at most that many results, capped at 10
3. IF the search request fails due to a network error, THEN THE Knowledge_Tool SHALL return a descriptive error message including the failure reason
4. IF no results match the query, THEN THE Knowledge_Tool SHALL return an empty results list with an informational message

### Requirement 4: Document Analysis Workflow Tool

**User Story:** As a Kiro user, I want to analyze a sample document to understand its structure and characteristics, so that I can make informed decisions about search strategy.

#### Acceptance Criteria

1. WHEN the `analyze_sample_document` tool is invoked with a JSON document string, THE Workflow_Tool SHALL return an analysis containing detected fields, data types, language hints, and content characteristics
2. WHEN the `analyze_sample_document` tool is invoked with plain text content, THE Workflow_Tool SHALL wrap the text in a `{"content": ...}` structure and analyze it
3. IF the document string is empty or contains only whitespace, THEN THE Workflow_Tool SHALL return an error message indicating the input is invalid
4. THE Workflow_Tool SHALL accept the document as an explicit parameter, not relying on global state

### Requirement 5: Search Strategy Recommendation Workflow Tool

**User Story:** As a Kiro user, I want to receive expert search strategy recommendations based on my requirements, so that I can design an optimal OpenSearch solution without interactive agent loops.

#### Acceptance Criteria

1. WHEN the `recommend_search_strategy` tool is invoked with a context string describing user requirements, THE Workflow_Tool SHALL return a structured recommendation containing retrieval method, algorithm, model selection, and deployment approach
2. THE Workflow_Tool SHALL internally use the Strands framework to invoke an LLM agent with access to the knowledge base tools for generating recommendations
3. WHEN the context includes document scale, language, latency, and budget information, THE Workflow_Tool SHALL incorporate all provided parameters into the recommendation
4. IF the internal agent call fails, THEN THE Workflow_Tool SHALL return a descriptive error message rather than raising an unhandled exception
5. THE Workflow_Tool SHALL return the recommendation in a single response without requiring interactive user input

### Requirement 6: Technical Q&A Workflow Tool

**User Story:** As a Kiro user, I want to ask technical questions about OpenSearch semantic search and receive expert answers, so that I can understand trade-offs and make informed decisions.

#### Acceptance Criteria

1. WHEN the `answer_opensearch_question` tool is invoked with a query string, THE Workflow_Tool SHALL return an expert answer grounded in the knowledge base content
2. THE Workflow_Tool SHALL internally use the Strands framework to invoke an LLM agent with access to the knowledge base tools and live documentation search
3. IF the internal agent call fails, THEN THE Workflow_Tool SHALL return a descriptive error message rather than raising an unhandled exception
4. THE Workflow_Tool SHALL return the answer in a single response without requiring interactive user input

### Requirement 7: Index Creation Operation Tool

**User Story:** As a Kiro user, I want to create OpenSearch indices with specified configurations, so that I can set up the search infrastructure as recommended.

#### Acceptance Criteria

1. WHEN the `create_index` tool is invoked with an index name and body configuration, THE Operation_Tool SHALL create the index in the OpenSearch cluster and return a success confirmation
2. WHEN the `create_index` tool is invoked with `replace_if_exists` set to true and the index already exists, THE Operation_Tool SHALL delete the existing index and create a new one
3. IF the OpenSearch cluster is unreachable, THEN THE Operation_Tool SHALL return a descriptive error message including connection details
4. IF the index body contains an invalid configuration, THEN THE Operation_Tool SHALL return the error response from OpenSearch
5. THE Operation_Tool SHALL accept OpenSearch connection parameters from environment variables (OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASSWORD)

### Requirement 8: Bedrock Embedding Model Deployment Tool

**User Story:** As a Kiro user, I want to register Bedrock embedding models in OpenSearch, so that I can use managed embedding services for vector search.

#### Acceptance Criteria

1. WHEN the `create_bedrock_embedding_model` tool is invoked with a model name, THE Operation_Tool SHALL register the model connector and deploy the model in OpenSearch
2. IF the model registration fails, THEN THE Operation_Tool SHALL return a descriptive error message including the failure reason from OpenSearch or AWS
3. THE Operation_Tool SHALL use AWS credentials from environment variables or the default credential chain for Bedrock API access

### Requirement 9: Local Pretrained Model Deployment Tool

**User Story:** As a Kiro user, I want to deploy local pretrained models in OpenSearch, so that I can use on-cluster models for sparse or dense vector search.

#### Acceptance Criteria

1. WHEN the `create_local_pretrained_model` tool is invoked with a model name, THE Operation_Tool SHALL register, deploy, and wait for the model to be ready in OpenSearch
2. IF the model deployment fails or times out, THEN THE Operation_Tool SHALL return a descriptive error message including the current model state

### Requirement 10: Pipeline Management Tool

**User Story:** As a Kiro user, I want to create and attach ingest or search pipelines to indices, so that I can configure document processing and query-time transformations.

#### Acceptance Criteria

1. WHEN the `create_and_attach_pipeline` tool is invoked with a pipeline name, index name, and pipeline body, THE Operation_Tool SHALL create the pipeline in OpenSearch and attach it to the specified index
2. WHEN the tool is invoked with `pipeline_type` set to "search", THE Operation_Tool SHALL attach the pipeline as a search pipeline using the `index.search.default_pipeline` setting
3. WHEN the tool is invoked with `is_hybrid_search` set to true and `hybrid_weights` provided, THE Operation_Tool SHALL generate a normalization-combination search pipeline with the specified weights
4. IF the pipeline creation or attachment fails, THEN THE Operation_Tool SHALL return a descriptive error message including the failure reason

### Requirement 11: Document Indexing Tool

**User Story:** As a Kiro user, I want to index documents into OpenSearch, so that I can populate indices with data for search.

#### Acceptance Criteria

1. WHEN the `index_doc` tool is invoked with an index name, document body, and document ID, THE Operation_Tool SHALL index the document and return a success confirmation
2. IF the indexing operation fails, THEN THE Operation_Tool SHALL return a descriptive error message including the failure reason

### Requirement 12: Document Deletion Tool

**User Story:** As a Kiro user, I want to delete documents from OpenSearch indices, so that I can manage index content.

#### Acceptance Criteria

1. WHEN the `delete_doc` tool is invoked with an index name and document ID, THE Operation_Tool SHALL delete the document and return a success confirmation
2. IF the document does not exist, THEN THE Operation_Tool SHALL return an appropriate message indicating the document was not found
3. IF the deletion operation fails, THEN THE Operation_Tool SHALL return a descriptive error message including the failure reason

### Requirement 13: Stateless Tool Design

**User Story:** As a developer, I want all MCP tools to be stateless with explicit parameters, so that the system is predictable and does not depend on hidden global state.

#### Acceptance Criteria

1. THE MCP_Server SHALL pass all required context as explicit Tool_Parameters to each tool invocation, not through global variables or module-level state
2. WHEN an Operation_Tool requires OpenSearch connection details, THE Operation_Tool SHALL read them from environment variables at invocation time
3. WHEN a Workflow_Tool requires user context or document data, THE Workflow_Tool SHALL accept the data as explicit string parameters

### Requirement 14: Kiro Power Packaging

**User Story:** As a Kiro user, I want the Power to be properly packaged with documentation and configuration, so that I can install and use it with minimal setup.

#### Acceptance Criteria

1. THE Kiro_Power SHALL include a `power.json` metadata file containing the Power name, description, version, and tool listing
2. THE Kiro_Power SHALL include a `POWER.md` file with user-facing documentation describing available tools and usage instructions
3. THE Kiro_Power SHALL include a `.kiro/settings/mcp.json` configuration file that specifies the MCP server command and required environment variables
4. THE Kiro_Power SHALL include a `requirements.txt` file listing all Python dependencies needed by the MCP server
5. THE Kiro_Power SHALL organize the MCP server code under an `mcp_server/` directory with separate modules for knowledge tools, workflow tools, and operation tools

### Requirement 15: Error Handling and Resilience

**User Story:** As a Kiro user, I want all tools to handle errors gracefully, so that I receive clear feedback when something goes wrong instead of cryptic failures.

#### Acceptance Criteria

1. WHEN any tool encounters an exception during execution, THE MCP_Server SHALL catch the exception and return a descriptive error string rather than propagating the exception to Kiro
2. WHEN an Operation_Tool cannot connect to OpenSearch, THE Operation_Tool SHALL include the host, port, and connection error details in the error message
3. WHEN a Workflow_Tool's internal Strands_Agent call fails, THE Workflow_Tool SHALL include the agent error details in the returned error message
