from opensearchpy import OpenSearch

from strands import tool
import os
import platform
import shutil
import subprocess
import time

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "myStrongPassword123!")
OPENSEARCH_DOCKER_IMAGE = os.getenv("OPENSEARCH_DOCKER_IMAGE", "opensearchproject/opensearch:latest")
OPENSEARCH_DOCKER_CONTAINER = os.getenv("OPENSEARCH_DOCKER_CONTAINER", "opensearch-local")
OPENSEARCH_DOCKER_START_TIMEOUT = int(os.getenv("OPENSEARCH_DOCKER_START_TIMEOUT", "120"))


def _build_client(use_ssl: bool) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    )


def _can_connect(opensearch_client: OpenSearch) -> bool:
    try:
        opensearch_client.info()
        return True
    except Exception:
        return False


def _is_local_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _run_docker_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _docker_install_hint() -> str:
    system_name = platform.system().lower()
    if system_name == "darwin":
        if shutil.which("brew"):
            return (
                "Install Docker Desktop with Homebrew: "
                "'brew install --cask docker && open -a Docker'. "
                "Official docs: https://docs.docker.com/desktop/setup/install/mac-install/"
            )
        return (
            "Install Docker Desktop for macOS: "
            "https://docs.docker.com/desktop/setup/install/mac-install/"
        )

    if system_name == "windows":
        return (
            "Install Docker Desktop for Windows: "
            "https://docs.docker.com/desktop/setup/install/windows-install/"
        )

    if system_name == "linux":
        return (
            "Install Docker Engine for Linux: "
            "https://docs.docker.com/engine/install/"
        )

    return (
        "Install Docker: https://docs.docker.com/get-started/get-docker/"
    )


def _docker_start_hint() -> str:
    system_name = platform.system().lower()
    if system_name in {"darwin", "windows"}:
        return "Start Docker Desktop and wait until it reports it is running."
    if system_name == "linux":
        return (
            "Start Docker service (for example: 'sudo systemctl start docker')."
        )
    return "Start the Docker daemon/service and retry."


def _start_local_opensearch_container() -> None:
    if not _is_local_host(OPENSEARCH_HOST):
        raise RuntimeError(
            f"Auto-start only supports local hosts. Current OPENSEARCH_HOST='{OPENSEARCH_HOST}'."
        )

    try:
        _run_docker_command(["docker", "--version"])
    except Exception as e:
        raise RuntimeError(
            "Docker is not installed or not available in PATH. "
            f"{_docker_install_hint()}"
        ) from e

    try:
        running = _run_docker_command(
            ["docker", "ps", "-q", "-f", f"name=^{OPENSEARCH_DOCKER_CONTAINER}$"]
        ).stdout.strip()
    except Exception as e:
        raise RuntimeError(
            "Docker CLI is available, but Docker daemon is not reachable. "
            f"{_docker_start_hint()}"
        ) from e
    if running:
        return

    existing = _run_docker_command(
        ["docker", "ps", "-aq", "-f", f"name=^{OPENSEARCH_DOCKER_CONTAINER}$"]
    ).stdout.strip()
    if existing:
        _run_docker_command(["docker", "rm", "-f", OPENSEARCH_DOCKER_CONTAINER])

    # Pull official OpenSearch Docker image and run a single-node instance.
    _run_docker_command(["docker", "pull", OPENSEARCH_DOCKER_IMAGE])
    _run_docker_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            OPENSEARCH_DOCKER_CONTAINER,
            "-p",
            f"{OPENSEARCH_PORT}:9200",
            "-p",
            "9600:9600",
            "-e",
            "discovery.type=single-node",
            "-e",
            "plugins.security.disabled=true",
            "-e",
            "DISABLE_INSTALL_DEMO_CONFIG=true",
            OPENSEARCH_DOCKER_IMAGE,
        ]
    )


def _wait_for_cluster_after_start() -> OpenSearch:
    secure_client = _build_client(use_ssl=True)
    insecure_client = _build_client(use_ssl=False)
    deadline = time.time() + OPENSEARCH_DOCKER_START_TIMEOUT

    while time.time() < deadline:
        if _can_connect(secure_client):
            return secure_client
        if _can_connect(insecure_client):
            return insecure_client
        time.sleep(2)

    raise RuntimeError(
        f"OpenSearch container did not become ready within {OPENSEARCH_DOCKER_START_TIMEOUT}s."
    )


def _create_client() -> OpenSearch:
    # First try secured localhost access (equivalent to:
    # curl https://localhost:9200 -u admin:myStrongPassword123! --insecure).
    secure_client = _build_client(use_ssl=True)
    if _can_connect(secure_client):
        return secure_client

    insecure_client = _build_client(use_ssl=False)
    if _can_connect(insecure_client):
        return insecure_client

    # Both direct connection attempts failed, so bootstrap local OpenSearch with Docker.
    _start_local_opensearch_container()
    return _wait_for_cluster_after_start()

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

def set_ml_settings(opensearch_client: OpenSearch | None = None):
    """Set the ML settings for the OpenSearch cluster.
    
    Returns:
        str: Success message or error.
    """
    if opensearch_client is None:
        opensearch_client = _create_client()

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
    opensearch_client.transport.perform_request("PUT", "/_cluster/settings", body=body)


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
        opensearch_client = _create_client()
        opensearch_client.indices.create(index=index_name, body=body)
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
        opensearch_client = _create_client()
        set_ml_settings(opensearch_client)
        response = opensearch_client.transport.perform_request("POST", "/_plugins/_ml/connectors/_create", body=connector_body)
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
        response = opensearch_client.transport.perform_request("POST", "/_plugins/_ml/models/_register", body=register_body)
        task_id = response.get("task_id")
        print(f"Model registration task started: {task_id}")
        
        # Poll for model ID
        model_id = None
        for _ in range(100):
            task_res = opensearch_client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
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
        response = opensearch_client.transport.perform_request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
        deploy_task_id = response.get("task_id")
        print(f"Model deployment task started: {deploy_task_id}")
        
        # Poll for deployment
        for _ in range(100):
            task_res = opensearch_client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{deploy_task_id}")
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
        opensearch_client = _create_client()
        if pipeline_type == "ingest":
            opensearch_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_body)
            settings = {"index.default_pipeline": pipeline_name}
        elif pipeline_type == "search":
            # Use low-level client for search pipeline to ensure compatibility
            opensearch_client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body)
            settings = {"index.search.default_pipeline": pipeline_name}
        else:
            return f"Error: Invalid pipeline_type '{pipeline_type}'. Must be 'ingest' or 'search'."

        opensearch_client.indices.put_settings(index=index_name, body=settings)
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
        opensearch_client = _create_client()
        # use red font to print the model_name
        print(f"\033[91m[create_local_pretrained_model] Model name: {model_name}\033[0m")
        if model_name not in PRETRAINED_MODELS:
            return f"Error: Model '{model_name}' not supported. Supported models: {list(PRETRAINED_MODELS.keys())}"
            
        model_version = PRETRAINED_MODELS[model_name]
        model_format = "TORCH_SCRIPT"
        
        set_ml_settings(opensearch_client)
        
        # 1. Register Model
        register_body = {
            "name": model_name,
            "version": model_version,
            "model_format": model_format
        }
        response = opensearch_client.transport.perform_request("POST", "/_plugins/_ml/models/_register", body=register_body)
        task_id = response.get("task_id")
        print(f"Model registration task started: {task_id}")
        
        # Poll for model ID
        model_id = None
        for _ in range(100):
            task_res = opensearch_client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
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
        response = opensearch_client.transport.perform_request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
        deploy_task_id = response.get("task_id")
        print(f"Model deployment task started: {deploy_task_id}")
        
        # Poll for deployment
        for _ in range(100):  # Increase timeout for deployment
            task_res = opensearch_client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{deploy_task_id}")
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
    opensearch_client = _create_client()
    try:
        opensearch_client.index(index=index_name, body=doc, id=doc_id)
    except Exception as e:
        return f"Failed to index document: {e}"

    opensearch_client.indices.refresh(index=index_name)

    try:
        return opensearch_client.get(index=index_name, id=doc_id)
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
        opensearch_client = _create_client()
        opensearch_client.delete(index=index_name, id=doc_id)
        return f"Document '{doc_id}' deleted from index '{index_name}' successfully."
    except Exception as e:
        return f"Failed to delete document: {e}"
