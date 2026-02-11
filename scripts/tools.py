import json
import re
from typing import Any, Dict, Optional
from html import unescape
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from strands import tool

_SAMPLE_DOC: Optional[Dict[str, Any]] = None


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
        filename = "scripts/knowledge/opensearch_semantic_search_guide.md"
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
        filename = "scripts/knowledge/dense_vector_models.md"
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
        filename = "scripts/knowledge/sparse_vector_models.md"
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading sparse vector models guide: {e}"

@tool
def get_sample_doc() -> str:
    """Get the stored sample document provided by the user.
    
    Returns:
        str: The sample document JSON string, or MISSING_SAMPLE_DOC if not set.
    """
    if _SAMPLE_DOC is None:
        return "MISSING_SAMPLE_DOC"
    return json.dumps(_SAMPLE_DOC, ensure_ascii=False)


@tool
def submit_sample_doc(doc: str) -> str:
    """Store a sample document provided by the user.

    Args:
        doc: User-provided sample document, preferably JSON.

    Returns:
        str: Status message indicating success or validation error.
    """
    global _SAMPLE_DOC

    raw = doc.strip()
    if not raw:
        return "Error: sample doc is empty."

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return "Error: sample doc must be a JSON object."
    except json.JSONDecodeError:
        # Fallback: treat plain text as content field.
        parsed = {"content": raw}

    _SAMPLE_DOC = parsed
    return "Sample document stored."

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _decode_duckduckgo_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return target
    return url


@tool
def search_opensearch_org(query: str, numberOfResults: int = 5) -> str:
    """Search from the OpenSearch documentation

    example query: "sparse_vector field parameters"

    This uses a site-restricted web query (`site:opensearch.org`) and returns
    the top matching results with title, URL, and snippet.

    Args:
        query: The query text to search for.
        numberOfResults: Max number of results to return.

    Returns:
        str: JSON string with query and filtered results from opensearch.org.
    """
    try:
        print(f"\033[91m[search_opensearch_org] Query: {query}\033[0m")

        limited_results = max(1, min(numberOfResults, 10))
        search_query = quote_plus(f"site:opensearch.org {query}")
        # DuckDuckGo's HTML endpoint is easy to fetch/parse without API keys or
        # anti-bot/captcha friction that commonly affects automated Google SERP scraping.
        url = f"https://duckduckgo.com/html/?q={search_query}"
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenSearchAgent/1.0)"},
        )

        with urlopen(request, timeout=15) as response:
            html = response.read().decode("utf-8", errors="ignore")

        title_matches = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|'
            r'<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippets = [left or right for left, right in snippet_matches]

        results = []
        for idx, (raw_href, raw_title) in enumerate(title_matches):
            href = _decode_duckduckgo_redirect(unescape(raw_href))
            netloc = urlparse(href).netloc.lower()
            if "opensearch.org" not in netloc:
                continue

            title = _normalize_text(unescape(_strip_html(raw_title)))
            snippet = ""
            if idx < len(snippets):
                snippet = _normalize_text(unescape(_strip_html(snippets[idx])))

            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                }
            )
            if len(results) >= limited_results:
                break

        if not results:
            return json.dumps(
                {
                    "query": query,
                    "results": [],
                    "message": "No opensearch.org results found.",
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "query": query,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return f"Error searching opensearch.org: {e}"
