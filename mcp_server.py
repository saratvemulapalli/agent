# /// script
# dependencies = ["mcp", "opensearch-py"]
# ///

from mcp.server.fastmcp import FastMCP

from scripts.tools import (
    submit_sample_doc,
    submit_sample_doc_from_local_file,
    submit_sample_doc_from_url,
    get_sample_docs_for_verification,
    read_knowledge_base,
    read_dense_vector_models,
    read_sparse_vector_models,
    search_opensearch_org,
)
from scripts.opensearch_ops_tools import (
    create_index,
    create_and_attach_pipeline,
    create_bedrock_embedding_model,
    create_local_pretrained_model,
    index_doc,
    index_verification_docs,
    delete_doc,
    cleanup_verification_docs,
    apply_capability_driven_verification,
    preview_capability_driven_verification,
    launch_search_ui,
    set_search_ui_suggestions,
)

mcp = FastMCP("OpenSearch Agent", json_response=True)

mcp.tool()(submit_sample_doc)
mcp.tool()(submit_sample_doc_from_local_file)
mcp.tool()(submit_sample_doc_from_url)
mcp.tool()(get_sample_docs_for_verification)
mcp.tool()(read_knowledge_base)
mcp.tool()(read_dense_vector_models)
mcp.tool()(read_sparse_vector_models)
mcp.tool()(search_opensearch_org)

mcp.tool()(create_index)
mcp.tool()(create_and_attach_pipeline)
mcp.tool()(create_bedrock_embedding_model)
mcp.tool()(create_local_pretrained_model)
mcp.tool()(index_doc)
mcp.tool()(index_verification_docs)
mcp.tool()(delete_doc)
mcp.tool()(cleanup_verification_docs)
mcp.tool()(apply_capability_driven_verification)
mcp.tool()(preview_capability_driven_verification)
mcp.tool()(launch_search_ui)
mcp.tool()(set_search_ui_suggestions)

if __name__ == "__main__":
    mcp.run(transport="stdio")
