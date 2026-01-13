from opensearchpy import OpenSearch, RequestsHttpConnection

from strands import tool
import json
import os
import time

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False,
)

PRETRAINED_MODELS = {
    "huggingface/cross-encoders/ms-marco-MiniLM-L-12-v2": "1.0.2",
    "huggingface/cross-encoders/ms-marco-MiniLM-L-6-v2": "1.0.2",
    "huggingface/sentence-transformers/all-MiniLM-L12-v2": "1.0.2",
    "huggingface/sentence-transformers/all-MiniLM-L6-v2": "1.0.2",
    "huggingface/sentence-transformers/all-distilroberta-v1": "1.0.2",
    "huggingface/sentence-transformers/all-mpnet-base-v2": "1.0.2",
    "huggingface/sentence-transformers/distiluse-base-multilingual-cased-v1": "1.0.2",
    "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b": "1.0.3",
    "huggingface/sentence-transformers/multi-qa-MiniLM-L6-cos-v1": "1.0.2",
    "huggingface/sentence-transformers/multi-qa-mpnet-base-dot-v1": "1.0.2",
    "huggingface/sentence-transformers/paraphrase-MiniLM-L3-v2": "1.0.2",
    "huggingface/sentence-transformers/paraphrase-mpnet-base-v2": "1.0.1",
    "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "1.0.2",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v1": "1.0.1",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v2-distill": "1.0.0",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v2-mini": "1.0.0",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-v2-distill": "1.0.0",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill": "1.0.0",
    "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-gte": "1.0.0",
    "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1": "1.0.1",
    "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-multilingual-v1": "1.0.0",
    "amazon/sentence-highlighting/opensearch-semantic-highlighter-v1": "1.0.0",
    "amazon/metrics_correlation": "1.0.0b2"
}

def set_ml_settings():
    """Set the ML settings for the OpenSearch cluster.
    
    Returns:
        str: Success message or error.
    """
    body = {
        "persistent":{
            "plugins.ml_commons.native_memory_threshold": 95,
            "plugins.ml_commons.only_run_on_ml_node": False,
            "plugins.ml_commons.allow_registering_model_via_url" : True,
            "plugins.ml_commons.model_access_control_enabled" : True,
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
                "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ]
        }
    }
    client.transport.perform_request("PUT", "/_cluster/settings", body=body)


@tool
def create_index(index_name: str, body: dict = None) -> str:
    """Create an OpenSearch index with the specified configuration.

    Args:
        index_name: The name of the index to create.
        body: The configuration body for the index (settings and mappings).

    Returns:
        str: Success message or error.
    """
    if body is None:
        body = {}
    try:
        client.indices.create(index=index_name, body=body)
        return f"Index '{index_name}' created successfully."
    except Exception as e:
        return f"Failed to create index '{index_name}': {e}"

@tool
def create_bedrock_embedding_model(model_name: str) -> str:
    """Create a Bedrock embedding model with the specified configuration.
    
    Args:
        model_name: The Bedrock model ID (e.g., "amazon.titan-embed-text-v2:0").
        
    Returns:
        str: The model ID of the created and deployed model, or error message.
    """
    if model_name != "amazon.titan-embed-text-v2:0":
        return "Error: Only amazon.titan-embed-text-v2:0 is supported for now."
    
    region = os.getenv("AWS_REGION", "us-east-1")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        return "Error: AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) are missing from environment variables."

    credentials = {
        "access_key": access_key,
        "secret_key": secret_key
    }
    if session_token:
        credentials["session_token"] = session_token

    # 1. Create Connector
    connector_body = {
        "name": f"Bedrock Connector for {model_name}",
        "description": f"Connector for Bedrock model {model_name}",
        "version": 1,
        "protocol": "aws_sigv4",
        "parameters": {
            "region": region,
            "service_name": "bedrock"
        },
        "credential": credentials,
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "url": f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_name}/invoke",
                "headers": {
                    "content-type": "application/json",
                    "x-amz-content-sha256": "required"
                },
                "request_body": "{ \"inputText\": \"${parameters.inputText}\", \"embeddingTypes\": [\"float\"] }",
                "pre_process_function": "connector.pre_process.bedrock.embedding",
                "post_process_function": "connector.post_process.bedrock_v2.embedding.float"
            }
        ]
    }
    
    try:
        set_ml_settings()
        response = client.transport.perform_request("POST", "/_plugins/_ml/connectors/_create", body=connector_body)
        connector_id = response.get("connector_id")
        if not connector_id:
            return f"Failed to create connector: {response}"
        print(f"Connector created: {connector_id}")

        # 2. Register Model
        register_body = {
            "name": f"Bedrock Model {model_name}",
            "function_name": "remote",
            "description": f"Bedrock embedding model {model_name}",
            "connector_id": connector_id
        }
        response = client.transport.perform_request("POST", "/_plugins/_ml/models/_register", body=register_body)
        task_id = response.get("task_id")
        print(f"Model registration task started: {task_id}")
        
        # Poll for model ID
        model_id = None
        for _ in range(100):
            task_res = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
            state = task_res.get("state")
            if state == "COMPLETED":
                model_id = task_res.get("model_id")
                break
            elif state == "FAILED":
                return f"Model registration failed: {task_res.get('error')}"
            time.sleep(2)
            
        if not model_id:
            return "Model registration timed out or failed."
        print(f"Model registered: {model_id}")

        # 3. Deploy Model
        response = client.transport.perform_request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
        deploy_task_id = response.get("task_id")
        print(f"Model deployment task started: {deploy_task_id}")
        
        # Poll for deployment
        for _ in range(100):
            task_res = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{deploy_task_id}")
            state = task_res.get("state")
            if state == "COMPLETED":
                return f"Model '{model_name}' (ID: {model_id}) created and deployed successfully."
            elif state == "FAILED":
                return f"Model deployment failed: {task_res.get('error')}"
            time.sleep(2)

        return f"Model deployment timed out. Model ID: {model_id}"

    except Exception as e:
        return f"Error creating Bedrock model: {e}"


@tool
def create_and_attach_pipeline(pipeline_name: str, pipeline_body: dict, index_name: str, pipeline_type: str = "ingest") -> str:
    """Create a pipeline (ingest or search) and attach it to an index.

    Usage Examples:
    1. Create and attach an ingest pipeline with dense and sparse vector embedding
    create_and_attach_pipeline("my_pipeline", {
        "processors": "processors" : [
            {
                "text_embedding": {
                    "model_id": "WKvvsJsB93bus0FT62-y",
                    "field_map": {
                        "text": "dense_embedding"
                    }
                }
            },
            {
                "sparse_encoding": {
                    "model_id": "wOB545cBlvLkxPs_jDLT",
                    "field_map": {
                        "text": "sparse_embedding"
                    }
                }
            }
        ]
    }, "my_index", "ingest")


    Args:
        pipeline_name: The name of the pipeline to create.
        pipeline_body: The configuration of the pipeline (processors, etc.).
        index_name: The name of the index to attach the pipeline to.
        pipeline_type: The type of pipeline, either 'ingest' or 'search'. Defaults to 'ingest'.

    Returns:
        str: Success message or error.
    """
    try:
        if pipeline_type == "ingest":
            client.ingest.put_pipeline(id=pipeline_name, body=pipeline_body)
            settings = {"index.default_pipeline": pipeline_name}
        elif pipeline_type == "search":
            # Use low-level client for search pipeline to ensure compatibility
            client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body)
            settings = {"index.search.default_pipeline": pipeline_name}
        else:
            return f"Error: Invalid pipeline_type '{pipeline_type}'. Must be 'ingest' or 'search'."

        client.indices.put_settings(index=index_name, body=settings)
        return f"{pipeline_type.capitalize()} pipeline '{pipeline_name}' created and attached to index '{index_name}' successfully."

    except Exception as e:
        return f"Failed to create and attach pipeline: {e}"


@tool
def create_local_pretrained_model(model_name: str) -> str:
    """Create a local pretrained model in OpenSearch.

    Usage Examples:
    1. Create and deploy a local pretrained model
    create_local_pretrained_model("amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-gte")
    2. Create and deploy a local pretrained tokenizer
    create_local_pretrained_model("amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1")

    Args:
        model_name: The name of the pretrained model (e.g., 'amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-gte').

    Returns:
        str: The model ID of the created and deployed model, or error message.
    """
    try:
        # use red font to print the model_name
        print(f"\033[91m[create_local_pretrained_model] Model name: {model_name}\033[0m")
        if model_name not in PRETRAINED_MODELS:
            return f"Error: Model '{model_name}' not supported. Supported models: {list(PRETRAINED_MODELS.keys())}"
            
        model_version = PRETRAINED_MODELS[model_name]
        model_format = "TORCH_SCRIPT"
        
        set_ml_settings()
        
        # 1. Register Model
        register_body = {
            "name": model_name,
            "version": model_version,
            "model_format": model_format
        }
        response = client.transport.perform_request("POST", "/_plugins/_ml/models/_register", body=register_body)
        task_id = response.get("task_id")
        print(f"Model registration task started: {task_id}")
        
        # Poll for model ID
        model_id = None
        for _ in range(100):
            task_res = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
            state = task_res.get("state")
            if state == "COMPLETED":
                model_id = task_res.get("model_id")
                break
            elif state == "FAILED":
                return f"Model registration failed: {task_res.get('error')}"
            time.sleep(5)
            
        if not model_id:
            return "Model registration timed out or failed."
        print(f"Model registered: {model_id}")

        # 2. Deploy Model
        response = client.transport.perform_request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
        deploy_task_id = response.get("task_id")
        print(f"Model deployment task started: {deploy_task_id}")
        
        # Poll for deployment
        for _ in range(100):  # Increase timeout for deployment
            task_res = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{deploy_task_id}")
            state = task_res.get("state")
            if state == "COMPLETED":
                return f"Model '{model_name}' (ID: {model_id}) created and deployed successfully."
            elif state == "FAILED":
                return f"Model deployment failed: {task_res.get('error')}"
            time.sleep(3)

        return f"Model deployment timed out. Model ID: {model_id}"

    except Exception as e:
        return f"Error creating local pretrained model: {e}"

@tool
def index_doc(index_name: str, doc: dict, doc_id: str) -> str:
    """Index a document into an OpenSearch index.

    Args:
        index_name: The name of the index to index the document into.
        doc: The document to index.
        doc_id: The ID of the document.

    Usage Examples:
    1. Index a document into an OpenSearch index
    index_doc("my_index", {"content": "The quick brown fox jumps over the lazy dog."}, "1")

    Returns:
        str: Document after ingest pipeline.
    """
    try:
        client.index(index=index_name, body=doc, id=doc_id)
    except Exception as e:
        return f"Failed to index document: {e}"

    client.indices.refresh(index=index_name)

    try:
        return client.get(index=index_name, id=doc_id)
    except Exception as e:
        return f"Failed to get document after ingest pipeline: {e}"

@tool
def delete_doc(index_name: str, doc_id: str) -> str:
    """Delete a document from an OpenSearch index.

    Args:
        index_name: The name of the index to delete the document from.
        doc_id: The ID of the document to delete.

    Returns:
        str: Success message or error.
    """
    try:
        client.delete(index=index_name, id=doc_id)
        return f"Document '{doc_id}' deleted from index '{index_name}' successfully."
    except Exception as e:
        return f"Failed to delete document: {e}"