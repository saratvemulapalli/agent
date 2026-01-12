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
        for _ in range(10):
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
        for _ in range(10):
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
