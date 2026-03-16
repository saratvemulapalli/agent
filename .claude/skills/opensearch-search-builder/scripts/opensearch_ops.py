#!/usr/bin/env python3
"""CLI wrapper for OpenSearch operations.

Subcommands expose the opensearch_orchestrator functions directly,
bypassing the MCP server and state machine. Outputs JSON to stdout.

Usage:
    uv run python scripts/opensearch_ops.py <command> [options]

Commands:
    status              Check OpenSearch connectivity
    create-index        Create an index with mappings
    deploy-model        Deploy a local pretrained ML model
    deploy-bedrock      Register a Bedrock embedding model
    create-pipeline     Create and attach an ingest/search pipeline
    index-doc           Index a single document
    index-bulk          Index verification docs from sample data
    launch-ui           Launch the Search Builder UI
    connect-ui          Connect Search UI to a remote endpoint
    search              Run a search query
    load-sample         Load sample data (file, URL, builtin IMDB)
    cleanup             Stop UI and clean up resources
"""

import argparse
import json
import sys
import os

# Ensure the repo root is on sys.path so opensearch_orchestrator is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def cmd_status(args):
    from opensearch_orchestrator.opensearch_ops_tools import _create_client, _can_connect
    try:
        client = _create_client()
        reachable, security = _can_connect(client)
        info = {"reachable": reachable, "security_enabled": security}
        if reachable:
            info["endpoint"] = f"http{'s' if security else ''}://localhost:9200"
        print(json.dumps(info))
    except Exception as e:
        print(json.dumps({"reachable": False, "error": str(e)}))


def cmd_create_index(args):
    from opensearch_orchestrator.opensearch_ops_tools import create_index
    body = json.loads(args.body) if args.body else {}
    result = create_index(
        index_name=args.name,
        body=body,
        replace_if_exists=args.replace,
    )
    print(result)


def cmd_deploy_model(args):
    from opensearch_orchestrator.opensearch_ops_tools import create_local_pretrained_model
    result = create_local_pretrained_model(model_name=args.name)
    print(result)


def cmd_deploy_bedrock(args):
    from opensearch_orchestrator.opensearch_ops_tools import create_bedrock_embedding_model
    result = create_bedrock_embedding_model(model_name=args.name)
    print(result)


def cmd_create_pipeline(args):
    from opensearch_orchestrator.opensearch_ops_tools import create_and_attach_pipeline
    body = json.loads(args.body) if args.body else {}
    result = create_and_attach_pipeline(
        pipeline_name=args.name,
        pipeline_body=body,
        index_name=args.index,
        pipeline_type=args.type,
        replace_if_exists=True,
        is_hybrid_search=args.hybrid,
        hybrid_weights=json.loads(args.weights) if args.weights else None,
    )
    print(result)


def cmd_index_doc(args):
    from opensearch_orchestrator.opensearch_ops_tools import index_doc
    doc = json.loads(args.doc)
    result = index_doc(index_name=args.index, doc=doc, doc_id=args.id)
    if isinstance(result, dict):
        print(json.dumps(result, default=str, ensure_ascii=False))
    else:
        print(result)


def cmd_index_bulk(args):
    from opensearch_orchestrator.opensearch_ops_tools import index_verification_docs
    result = index_verification_docs(
        index_name=args.index,
        count=args.count,
        source_local_file=args.source_file or "",
        source_index_name=args.source_index or "",
    )
    print(result)


def cmd_launch_ui(args):
    from opensearch_orchestrator.opensearch_ops_tools import launch_search_ui
    result = launch_search_ui(index_name=args.index or "")
    print(result)


def cmd_connect_ui(args):
    from opensearch_orchestrator.opensearch_ops_tools import connect_search_ui_to_endpoint
    result = connect_search_ui_to_endpoint(
        endpoint=args.endpoint,
        port=args.port,
        use_ssl=not args.no_ssl,
        username=args.username or "",
        password=args.password or "",
        aws_region=args.aws_region or "",
        aws_service=args.aws_service or "",
        index_name=args.index or "",
    )
    print(result)


def cmd_search(args):
    from opensearch_orchestrator.opensearch_ops_tools import _create_client
    client = _create_client()
    body = json.loads(args.body) if args.body else {"query": {"match_all": {}}}
    result = client.search(index=args.index, body=body, size=args.size)
    print(json.dumps(result, default=str, ensure_ascii=False, indent=2))


def cmd_load_sample(args):
    from opensearch_orchestrator.tools import (
        submit_sample_doc,
        submit_sample_doc_from_local_file,
        submit_sample_doc_from_url,
        submit_sample_doc_from_localhost_index,
    )
    if args.source_type == "builtin_imdb":
        result = submit_sample_doc_from_local_file(
            "opensearch_orchestrator/sample_data/imdb.title.basics.tsv"
        )
    elif args.source_type == "local_file":
        result = submit_sample_doc_from_local_file(args.source_value)
    elif args.source_type == "url":
        result = submit_sample_doc_from_url(args.source_value)
    elif args.source_type == "localhost_index":
        result = submit_sample_doc_from_localhost_index(args.source_value or "")
    elif args.source_type == "paste":
        result = submit_sample_doc(args.source_value)
    else:
        result = f"Unknown source_type: {args.source_type}"
    print(result)


def cmd_cleanup(args):
    from opensearch_orchestrator.opensearch_ops_tools import cleanup_ui_server
    result = cleanup_ui_server()
    print(result)


def cmd_read_knowledge(args):
    """Read a knowledge base file and print its content."""
    knowledge_dir = os.path.join(
        os.path.dirname(__file__), "..", "references", "knowledge"
    )
    target = os.path.join(knowledge_dir, args.file)
    if not os.path.isfile(target):
        available = os.listdir(knowledge_dir) if os.path.isdir(knowledge_dir) else []
        print(f"File not found: {args.file}. Available: {available}", file=sys.stderr)
        sys.exit(1)
    with open(target) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="OpenSearch operations CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Check OpenSearch connectivity")

    # create-index
    p = sub.add_parser("create-index", help="Create an index")
    p.add_argument("--name", required=True, help="Index name")
    p.add_argument("--body", default="{}", help="JSON index body (settings + mappings)")
    p.add_argument("--replace", action="store_true", default=True, help="Replace if exists")

    # deploy-model
    p = sub.add_parser("deploy-model", help="Deploy a local pretrained model")
    p.add_argument("--name", required=True, help="Model name (e.g. huggingface/sentence-transformers/all-MiniLM-L6-v2)")

    # deploy-bedrock
    p = sub.add_parser("deploy-bedrock", help="Register a Bedrock embedding model")
    p.add_argument("--name", required=True, help="Bedrock model ID (e.g. amazon.titan-embed-text-v2:0)")

    # create-pipeline
    p = sub.add_parser("create-pipeline", help="Create and attach a pipeline")
    p.add_argument("--name", required=True, help="Pipeline name")
    p.add_argument("--body", required=True, help="JSON pipeline body")
    p.add_argument("--index", required=True, help="Target index name")
    p.add_argument("--type", default="ingest", choices=["ingest", "search"], help="Pipeline type")
    p.add_argument("--hybrid", action="store_true", help="Hybrid search pipeline")
    p.add_argument("--weights", default=None, help="JSON array of hybrid weights [lexical, semantic]")

    # index-doc
    p = sub.add_parser("index-doc", help="Index a single document")
    p.add_argument("--index", required=True, help="Target index")
    p.add_argument("--doc", required=True, help="JSON document")
    p.add_argument("--id", required=True, help="Document ID")

    # index-bulk
    p = sub.add_parser("index-bulk", help="Index verification docs from sample data")
    p.add_argument("--index", required=True, help="Target index")
    p.add_argument("--count", type=int, default=10, help="Number of docs (max 100)")
    p.add_argument("--source-file", default=None, help="Local file path for sample data")
    p.add_argument("--source-index", default=None, help="Localhost index for sample data")

    # launch-ui
    p = sub.add_parser("launch-ui", help="Launch Search Builder UI")
    p.add_argument("--index", default="", help="Default index name")

    # connect-ui
    p = sub.add_parser("connect-ui", help="Connect UI to remote endpoint")
    p.add_argument("--endpoint", required=True, help="OpenSearch host")
    p.add_argument("--port", type=int, default=443, help="Port (default 443)")
    p.add_argument("--no-ssl", action="store_true", help="Disable SSL")
    p.add_argument("--username", default="", help="Username for auth")
    p.add_argument("--password", default="", help="Password for auth")
    p.add_argument("--aws-region", default="", help="AWS region for SigV4")
    p.add_argument("--aws-service", default="", help="AWS service (aoss or es)")
    p.add_argument("--index", default="", help="Default index name")

    # search
    p = sub.add_parser("search", help="Run a search query")
    p.add_argument("--index", required=True, help="Index to search")
    p.add_argument("--body", default=None, help="JSON search body")
    p.add_argument("--size", type=int, default=10, help="Number of results")

    # load-sample
    p = sub.add_parser("load-sample", help="Load sample documents")
    p.add_argument("--source-type", required=True,
                    choices=["builtin_imdb", "local_file", "url", "localhost_index", "paste"],
                    help="Source type")
    p.add_argument("--source-value", default="", help="File path, URL, index name, or JSON doc")

    # cleanup
    sub.add_parser("cleanup", help="Stop UI and clean up")

    # read-knowledge
    p = sub.add_parser("read-knowledge", help="Read a knowledge base file")
    p.add_argument("--file", required=True, help="Filename (e.g. dense_vector_models.md)")

    args = parser.parse_args()

    dispatch = {
        "status": cmd_status,
        "create-index": cmd_create_index,
        "deploy-model": cmd_deploy_model,
        "deploy-bedrock": cmd_deploy_bedrock,
        "create-pipeline": cmd_create_pipeline,
        "index-doc": cmd_index_doc,
        "index-bulk": cmd_index_bulk,
        "launch-ui": cmd_launch_ui,
        "connect-ui": cmd_connect_ui,
        "search": cmd_search,
        "load-sample": cmd_load_sample,
        "cleanup": cmd_cleanup,
        "read-knowledge": cmd_read_knowledge,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
