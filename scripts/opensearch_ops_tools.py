from opensearchpy import OpenSearch

from strands import tool
from scripts.shared import normalize_text, value_shape, text_richness_score
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scripts.tools import get_sample_docs_payload, set_ingest_source_field_hints

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "myStrongPassword123!")
OPENSEARCH_DOCKER_IMAGE = os.getenv("OPENSEARCH_DOCKER_IMAGE", "opensearchproject/opensearch:latest")
OPENSEARCH_DOCKER_CONTAINER = os.getenv("OPENSEARCH_DOCKER_CONTAINER", "opensearch-local")
OPENSEARCH_DOCKER_START_TIMEOUT = int(os.getenv("OPENSEARCH_DOCKER_START_TIMEOUT", "120"))
SEARCH_UI_HOST = os.getenv("SEARCH_UI_HOST", "127.0.0.1")
SEARCH_UI_PORT = int(os.getenv("SEARCH_UI_PORT", "8765"))
SEARCH_UI_STATIC_DIR = (
    Path(__file__).resolve().parent / "ui" / "search_builder"
)

_SEARCH_UI_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".jsx": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}

_VERIFICATION_DOC_TRACKER: dict[str, list[str]] = {}
_LAST_VERIFICATION_INDEX: str = ""
_SEARCH_UI_SERVER: ThreadingHTTPServer | None = None
_SEARCH_UI_SERVER_THREAD: threading.Thread | None = None
_SEARCH_UI_DEFAULT_INDEX: str = ""
_SEARCH_UI_SERVER_LOCK = threading.Lock()
_SEARCH_UI_SUGGESTION_META_BY_INDEX: dict[str, list[dict[str, object]]] = {}


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


def _normalize_text(value: object) -> str:
    """Normalize any value into a compact single-line string with collapsed whitespace.

    Delegates to the shared ``normalize_text`` utility.
    """
    return normalize_text(value)


def _normalized_query_key(value: object) -> str:
    return _normalize_text(value).lower()


def _canonical_capability_id(label: str) -> str:
    lowered = label.lower()
    if "exact" in lowered:
        return "exact"
    if "semantic" in lowered:
        return "semantic"
    if "structured" in lowered or "filter" in lowered:
        return "structured"
    if "combined" in lowered:
        return "combined"
    if "autocomplete" in lowered or "prefix" in lowered:
        return "autocomplete"
    if "fuzzy" in lowered or "typo" in lowered:
        return "fuzzy"
    return ""


def _extract_search_capabilities(worker_output: str) -> list[dict[str, object]]:
    """Parse the worker's markdown output and extract the "Search Capabilities" section.

    The worker output is expected to contain a bullet list under a heading that
    includes the phrase "Search Capabilities". Each capability bullet must use
    a canonical prefix (case-insensitive): Exact:, Semantic:, Structured:,
    Combined:, Autocomplete:, or Fuzzy:.

    Args:
        worker_output: Raw markdown text produced by the worker.

    Returns:
        A list of capability dicts, each with the following keys:

        - ``"id"`` (str): Canonical capability identifier derived from the
          bullet text (e.g. ``"exact"``, ``"semantic"``, ``"structured"``,
          ``"combined"``, ``"autocomplete"``, ``"fuzzy"``).
        - ``"label"`` (str): The normalized human-readable text of the bullet
          point (e.g. ``"exact match toyota camry honda civic"``).
        - ``"examples"`` (list[str]): Capability examples populated from
          selected sample documents later in the verification flow.

    Example::

        # If the worker output contains:
        #   ## Search Capabilities
        #   - Exact match: "Toyota Camry", "Honda Civic"
        #   - Semantic search: "fuel efficient family car"
        #
        # The return value would be:
        [
            {
                "id": "exact",
                "label": "exact match toyota camry honda civic",
                "examples": [],
            },
            {
                "id": "semantic",
                "label": "semantic search fuel efficient family car",
                "examples": [],
            },
        ]
    """
    if not worker_output:
        return []

    capabilities: list[dict[str, object]] = []
    seen: set[str] = set()
    in_section = False

    for raw_line in worker_output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()

        if not in_section and "search capabilities" in lowered:
            in_section = True
            continue

        if not in_section:
            continue

        if not line:
            if capabilities:
                break
            continue

        if line.startswith("##") or line.startswith("---"):
            break

        if not (line.startswith("-") or line.startswith("*")):
            if capabilities:
                break
            continue

        bullet = re.sub(r"^[-*]\s*", "", line)
        bullet = re.sub(r"^[^\w]+", "", bullet)
        prefix_match = re.match(
            r"^(exact|semantic|structured|combined|autocomplete|fuzzy)\s*:",
            bullet,
            re.IGNORECASE,
        )
        if not prefix_match:
            continue

        capability_id = _canonical_capability_id(prefix_match.group(1))
        if not capability_id or capability_id in seen:
            continue

        capabilities.append(
            {
                "id": capability_id,
                "label": _normalize_text(bullet),
                "examples": [],
            }
        )
        seen.add(capability_id)

    return capabilities


def _extract_index_field_specs(opensearch_client: OpenSearch, index_name: str) -> dict[str, dict[str, str]]:
    field_specs: dict[str, dict[str, str]] = {}
    try:
        mapping_response = opensearch_client.indices.get_mapping(index=index_name)
    except Exception:
        return field_specs

    index_mapping = {}
    if isinstance(mapping_response, dict):
        index_mapping = next(iter(mapping_response.values()), {})
    mappings = index_mapping.get("mappings", {})

    def _walk(properties: dict, prefix: str = "") -> None:
        if not isinstance(properties, dict):
            return
        for field_name, config in properties.items():
            if not isinstance(config, dict):
                continue
            full_name = f"{prefix}.{field_name}" if prefix else field_name

            field_type = config.get("type")
            if isinstance(field_type, str):
                field_specs[full_name] = {
                    "type": field_type,
                    "normalizer": str(config.get("normalizer", "")).strip(),
                }

            sub_fields = config.get("fields")
            if isinstance(sub_fields, dict):
                for sub_name, sub_config in sub_fields.items():
                    if not isinstance(sub_config, dict):
                        continue
                    sub_type = sub_config.get("type")
                    if not isinstance(sub_type, str):
                        continue
                    field_specs[f"{full_name}.{sub_name}"] = {
                        "type": sub_type,
                        "normalizer": str(sub_config.get("normalizer", "")).strip(),
                    }

            nested_props = config.get("properties")
            if isinstance(nested_props, dict):
                _walk(nested_props, full_name)

    _walk(mappings.get("properties", {}))
    return field_specs


def _extract_declared_field_types_from_index_body(body: dict) -> dict[str, str]:
    declared_field_types: dict[str, str] = {}
    if not isinstance(body, dict):
        return declared_field_types

    mappings = body.get("mappings")
    if not isinstance(mappings, dict):
        return declared_field_types

    properties = mappings.get("properties")
    if not isinstance(properties, dict):
        return declared_field_types

    def _walk(props: dict, prefix: str = "") -> None:
        if not isinstance(props, dict):
            return
        for raw_name, config in props.items():
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if not name or not isinstance(config, dict):
                continue

            full_name = f"{prefix}.{name}" if prefix else name
            field_type = str(config.get("type", "")).strip().lower()
            if field_type:
                declared_field_types[full_name] = field_type

            sub_fields = config.get("fields")
            if isinstance(sub_fields, dict):
                for raw_sub_name, sub_config in sub_fields.items():
                    if not isinstance(raw_sub_name, str) or not isinstance(sub_config, dict):
                        continue
                    sub_name = raw_sub_name.strip()
                    if not sub_name:
                        continue
                    sub_type = str(sub_config.get("type", "")).strip().lower()
                    if sub_type:
                        declared_field_types[f"{full_name}.{sub_name}"] = sub_type

            nested_properties = config.get("properties")
            if isinstance(nested_properties, dict):
                _walk(nested_properties, full_name)

    _walk(properties)
    return declared_field_types


def _normalize_knn_method_engines(index_body: dict) -> list[str]:
    """Normalize deprecated/implicit k-NN engines to stable defaults.

    This guardrail prevents LLM-generated index bodies from relying on deprecated
    ``nmslib`` or implicit engine defaults that vary by OpenSearch version.
    """
    if not isinstance(index_body, dict):
        return []

    mappings = index_body.get("mappings")
    if not isinstance(mappings, dict):
        return []

    properties = mappings.get("properties")
    if not isinstance(properties, dict):
        return []

    updates: list[str] = []

    def _preferred_engine(method_name: str, current_engine: str) -> str:
        method = method_name.strip().lower()
        engine = current_engine.strip().lower()

        if method == "ivf":
            return "faiss"
        if method == "hnsw":
            if not engine or engine == "nmslib":
                return "lucene"
            return engine
        if engine == "nmslib":
            return "faiss"
        return engine

    def _walk(props: dict, prefix: str = "") -> None:
        if not isinstance(props, dict):
            return
        for raw_name, config in props.items():
            if not isinstance(raw_name, str) or not isinstance(config, dict):
                continue

            name = raw_name.strip()
            if not name:
                continue

            full_name = f"{prefix}.{name}" if prefix else name
            field_type = str(config.get("type", "")).strip().lower()

            if field_type == "knn_vector":
                method = config.get("method")
                if isinstance(method, dict):
                    method_name = str(method.get("name", "")).strip().lower()
                    current_engine = str(method.get("engine", "")).strip().lower()
                    next_engine = _preferred_engine(method_name, current_engine)
                    if next_engine and next_engine != current_engine:
                        method["engine"] = next_engine
                        before = current_engine or "<empty>"
                        updates.append(
                            f"{full_name}: engine {before} -> {next_engine}"
                        )

            nested_properties = config.get("properties")
            if isinstance(nested_properties, dict):
                _walk(nested_properties, full_name)

    _walk(properties)
    return updates


def _collect_requested_vs_existing_field_type_mismatches(
    requested_field_types: dict[str, str],
    existing_field_types: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    if not isinstance(requested_field_types, dict) or not requested_field_types:
        return mismatches
    if not isinstance(existing_field_types, dict):
        existing_field_types = {}

    normalized_existing = {
        str(field_name).strip(): str(field_type).strip().lower()
        for field_name, field_type in existing_field_types.items()
        if str(field_name).strip() and str(field_type).strip()
    }

    def _resolve_existing_type(requested_field_name: str) -> str:
        if requested_field_name in normalized_existing:
            return normalized_existing[requested_field_name]

        requested_lower = requested_field_name.lower()
        for candidate_name, candidate_type in normalized_existing.items():
            if candidate_name.lower() == requested_lower:
                return candidate_type

        requested_leaf = requested_field_name.split(".")[-1].lower()
        leaf_matches = [
            candidate_type
            for candidate_name, candidate_type in normalized_existing.items()
            if candidate_name.split(".")[-1].lower() == requested_leaf
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]

        return ""

    def _types_compatible(requested_type: str, existing_type: str) -> bool:
        if requested_type == existing_type:
            return True

        compatibility_groups = [
            {"keyword", "constant_keyword"},
            {"text", "match_only_text"},
            {"byte", "short", "integer", "long"},
            {"half_float", "float", "double", "scaled_float"},
        ]
        for group in compatibility_groups:
            if requested_type in group and existing_type in group:
                return True
        return False

    for requested_field_name, requested_type in sorted(requested_field_types.items()):
        normalized_requested_field = str(requested_field_name).strip()
        normalized_requested_type = str(requested_type).strip().lower()
        if not normalized_requested_field or not normalized_requested_type:
            continue

        existing_type = _resolve_existing_type(normalized_requested_field)
        if not existing_type:
            mismatches.append(
                f"Field '{normalized_requested_field}' is missing in existing index (requested type '{normalized_requested_type}')."
            )
            continue

        if not _types_compatible(normalized_requested_type, existing_type):
            mismatches.append(
                f"Field '{normalized_requested_field}' requested type '{normalized_requested_type}' but existing type is '{existing_type}'."
            )

    return mismatches


def _resolve_field_spec_for_doc_key(
    field_name: str, field_specs: dict[str, dict[str, str]]
) -> tuple[str, dict[str, str]]:
    if field_name in field_specs:
        return field_name, field_specs[field_name]

    lowered = field_name.lower()
    for candidate_name, candidate_spec in field_specs.items():
        if candidate_name.lower() == lowered:
            return candidate_name, candidate_spec

    for candidate_name, candidate_spec in field_specs.items():
        if candidate_name.split(".")[-1].lower() == lowered:
            return candidate_name, candidate_spec

    return "", {}

def _value_shape(text: str) -> dict[str, object]:
    """Compute structural characteristics of a text value.

    Delegates to the shared ``value_shape`` utility.
    """
    return value_shape(text)


def _extract_doc_features(
    source: dict, field_specs: dict[str, dict[str, str]]
) -> dict[str, object]:
    """Analyse a single document and classify its fields into search-demo categories.

    Each scalar (non-null, non-nested) field in *source* is inspected against
    the index mapping (``field_specs``) and its value shape to determine which
    types of search queries it can demonstrate.

    Args:
        source: The raw document dict as returned from OpenSearch
            (e.g. ``{"title": "Toyota Camry", "price": 25000, ...}``).
        field_specs: Mapping of field path to its index mapping metadata,
            as returned by ``_extract_index_field_specs``.  Each value is a
            dict with at least ``"type"`` and optionally ``"normalizer"``.

    Returns:
        A features dict with the following keys:

        - ``"source"`` (dict): The original raw document, kept as-is.
        - ``"scalar_items"`` (list[dict]): All non-null, non-nested fields
          with their resolved mapping info and value shape.  Each entry has:

          - ``"key"`` (str): The original field key in the document.
          - ``"field"`` (str): The resolved mapping field path.
          - ``"type"`` (str): The mapping type (e.g. ``"text"``, ``"keyword"``,
            ``"integer"``).
          - ``"normalizer"`` (str): The normalizer name, if any.
          - ``"shape"`` (dict): Value shape returned by ``_value_shape``.

        - ``"exact_candidates"`` (list[dict]): Fields/values suitable for
          exact-match (``term``) or ``match_phrase`` queries.  Each entry has:

          - ``"text"`` (str): The normalised value text.
          - ``"query_mode"`` (str): ``"term"`` or ``"match_phrase"``.
          - ``"field"`` (str): The field to query against.
          - ``"case_insensitive"`` (bool): Whether a case-insensitive match
            is available (based on the normalizer).

        - ``"phrase_candidates"`` (list[dict]): Fields/values suitable for
          ``match_phrase`` queries (text-type fields).  Same keys as
          ``exact_candidates``.
        - ``"semantic_candidates"`` (list[dict]): Fields/values suitable for
          semantic / neural search (multi-token, mostly alphabetic text).
          Each entry has:

          - ``"text"`` (str): The normalised value text.
          - ``"field"`` (str): The field to query against.

        - ``"structured_candidates"`` (list[dict]): Fields/values suitable for
          structured queries (numeric, date, boolean, keyword).  Each entry has:

          - ``"field"`` (str): The field to query against.
          - ``"value"`` (str): The normalised value text.
          - ``"type"`` (str): The mapping type.

        - ``"anchor_tokens"`` (list[dict]): Individual tokens (>= 2 chars,
          non-digit) extracted from all scalar values, with source field info.
          Each entry has:

          - ``"token"`` (str): The token text.
          - ``"field"`` (str): The source field where the token came from.

    Example::

        # For a document:
        #   {"title": "Toyota Camry", "price": 25000,
        #    "description": "A reliable family sedan"}
        #
        # A possible return value:
        {
            "source": {"title": "Toyota Camry", "price": 25000,
                        "description": "A reliable family sedan"},
            "scalar_items": [
                {"key": "title", "field": "title", "type": "text",
                 "normalizer": "",
                 "shape": {"text": "toyota camry", "token_count": 2,
                           "alpha_ratio": 0.92, ...}},
                {"key": "price", "field": "price", "type": "integer",
                 "normalizer": "",
                 "shape": {"text": "25000", "token_count": 1,
                           "looks_numeric": True, ...}},
                {"key": "description", "field": "description", "type": "text",
                 "normalizer": "",
                 "shape": {"text": "a reliable family sedan",
                           "token_count": 4, "alpha_ratio": 0.95, ...}},
            ],
            "exact_candidates": [
                {"text": "toyota camry", "query_mode": "term",
                 "field": "title.keyword", "case_insensitive": False},
            ],
            "phrase_candidates": [
                {"text": "toyota camry", "query_mode": "match_phrase",
                 "field": "title", "case_insensitive": False},
                {"text": "a reliable family sedan",
                 "query_mode": "match_phrase", "field": "description",
                 "case_insensitive": False},
            ],
            "semantic_candidates": [
                {"text": "a reliable family sedan",
                 "field": "description"},
            ],
            "structured_candidates": [
                {"field": "price", "value": "25000", "type": "integer"},
            ],
            "anchor_tokens": [
                {"token": "toyota", "field": "title"},
                {"token": "camry", "field": "title"},
                {"token": "reliable", "field": "description"},
                {"token": "family", "field": "description"},
                {"token": "sedan", "field": "description"},
            ],
        }
    """
    scalar_items: list[dict[str, object]] = []
    for key, value in source.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        compact = _normalize_text(value)
        if not compact:
            continue

        resolved_field, resolved_spec = _resolve_field_spec_for_doc_key(str(key), field_specs)
        shape = _value_shape(compact)
        scalar_items.append(
            {
                "key": str(key),
                "field": resolved_field,
                "type": resolved_spec.get("type", ""),
                "normalizer": resolved_spec.get("normalizer", ""),
                "shape": shape,
            }
        )

    exact_candidates: list[dict[str, object]] = []
    phrase_candidates: list[dict[str, object]] = []
    semantic_candidates: list[dict[str, object]] = []
    structured_candidates: list[dict[str, object]] = []
    anchor_tokens: list[dict[str, str]] = []
    keyword_types = {"keyword", "constant_keyword"}
    structured_types = {"byte", "short", "integer", "long", "float", "half_float", "double", "scaled_float", "date", "boolean", "keyword", "constant_keyword"}

    for item in scalar_items:
        key = str(item["key"])
        field = str(item["field"])
        field_type = str(item["type"])
        normalizer = str(item["normalizer"])
        shape = item["shape"]
        text_value = str(shape["text"])
        token_count = int(shape["token_count"])
        alpha_ratio = float(shape["alpha_ratio"])
        looks_numeric = bool(shape["looks_numeric"])
        looks_date = bool(shape["looks_date"])

        if field_type in keyword_types and 2 <= len(text_value) <= 180:
            exact_candidates.append(
                {
                    "text": text_value,
                    "query_mode": "term",
                    "field": field or key,
                    "case_insensitive": bool(normalizer),
                }
            )

        if field_type == "text":
            keyword_alias = f"{field}.keyword" if field else ""
            keyword_spec = field_specs.get(keyword_alias, {})
            if keyword_spec.get("type", "") in keyword_types and 2 <= len(text_value) <= 180:
                exact_candidates.append(
                    {
                        "text": text_value,
                        "query_mode": "term",
                        "field": keyword_alias,
                        "case_insensitive": bool(keyword_spec.get("normalizer", "")),
                    }
                )
            if token_count >= 1 and 2 <= len(text_value) <= 220:
                phrase_candidates.append(
                    {
                        "text": text_value,
                        "query_mode": "match_phrase",
                        "field": field or key,
                        "case_insensitive": False,
                    }
                )

        if token_count >= 2 and alpha_ratio >= 0.35 and 8 <= len(text_value) <= 220:
            semantic_candidates.append(
                {
                    "text": text_value,
                    "field": field or key,
                }
            )

        if (
            field_type in structured_types
            or looks_numeric
            or looks_date
            or field_type == "boolean"
        ):
            structured_candidates.append(
                {
                    "field": field or key,
                    "value": text_value,
                    "type": field_type,
                }
            )

        for token in shape["tokens"]:
            if token.isdigit():
                continue
            if len(token) >= 2:
                anchor_tokens.append({"token": token, "field": field or key})

    if not exact_candidates:
        for candidate in phrase_candidates:
            exact_candidates.append(
                {
                    "text": candidate["text"],
                    "query_mode": "match_phrase",
                    "field": candidate["field"],
                    "case_insensitive": False,
                }
            )
            break

    if not semantic_candidates:
        for item in scalar_items:
            shape = item["shape"]
            text_value = str(shape["text"])
            if len(text_value) >= 4:
                semantic_candidates.append(
                    {
                        "text": text_value,
                        "field": item["field"] or item["key"],
                    }
                )
                break

    return {
        "source": source,
        "scalar_items": scalar_items,
        "exact_candidates": exact_candidates,
        "phrase_candidates": phrase_candidates,
        "semantic_candidates": semantic_candidates,
        "structured_candidates": structured_candidates,
        "anchor_tokens": anchor_tokens,
    }


def _anchor_token_text(entry: object) -> str:
    if isinstance(entry, dict):
        return _normalize_text(entry.get("token", ""))
    return _normalize_text(entry)


def _anchor_token_field(entry: object) -> str:
    if isinstance(entry, dict):
        return _normalize_text(entry.get("field", ""))
    return ""


def _first_anchor_token(anchor_tokens: list[object], min_len: int) -> tuple[str, str]:
    for entry in anchor_tokens:
        token = _anchor_token_text(entry)
        if len(token) >= min_len:
            return token, _anchor_token_field(entry)
    return "", ""


def _score_doc_for_capability(features: dict[str, object], capability_id: str) -> float:
    exact_candidates = features.get("exact_candidates", [])
    semantic_candidates = features.get("semantic_candidates", [])
    structured_candidates = features.get("structured_candidates", [])
    anchor_tokens = features.get("anchor_tokens", [])
    if not isinstance(anchor_tokens, list):
        anchor_tokens = []

    if capability_id == "exact":
        if not exact_candidates:
            return 0.0
        best = next(iter(exact_candidates), {})
        mode = str(best.get("query_mode", ""))
        mode_bonus = 120.0 if mode == "term" else 80.0
        return mode_bonus + min(len(str(best.get("text", ""))), 100) / 100.0

    if capability_id == "semantic":
        if not semantic_candidates:
            return 0.0
        best_text = _best_semantic_text_from_candidates(semantic_candidates)
        if not best_text:
            return 0.0
        return 60.0 + min(len(best_text), 200) / 10.0

    if capability_id == "structured":
        return float(len(structured_candidates) * 20)

    if capability_id == "combined":
        if not semantic_candidates or not structured_candidates:
            return 0.0
        best_text = _best_semantic_text_from_candidates(semantic_candidates)
        if not best_text:
            return 0.0
        return 80.0 + float(len(structured_candidates)) + min(len(best_text), 200) / 20.0

    if capability_id == "autocomplete":
        longest = max((len(_anchor_token_text(token)) for token in anchor_tokens), default=0)
        return float(longest if longest >= 3 else 0)

    if capability_id == "fuzzy":
        longest = max((len(_anchor_token_text(token)) for token in anchor_tokens), default=0)
        return float(40 + longest) if longest >= 5 else 0.0

    return 0.0


def _select_docs_by_capability(
    features_list: list[dict[str, object]],
    capabilities: list[dict[str, object]],
) -> tuple[dict[str, int], list[str]]:
    """Assign the best sample document to each capability using a "prefer unique, allow reuse" strategy.

    For every capability the function scores each document (via
    ``_score_doc_for_capability``) and picks the highest-scoring one that has
    **not** already been claimed by a previous capability.  If all compatible
    documents are already taken, the highest-scoring document is reused as a
    fallback so the capability still has a demo document.

    Args:
        features_list: One entry per candidate document, as returned by
            ``_extract_doc_features``.  Each dict contains categorised field
            candidates (``exact_candidates``, ``semantic_candidates``,
            ``structured_candidates``, ``anchor_tokens``, etc.) that
            ``_score_doc_for_capability`` uses to compute a relevance score.

            See ``_extract_doc_features`` for the full schema and an example.

        capabilities: One entry per search capability, as returned by
            ``_extract_search_capabilities``.  Each dict has the following keys:

            - ``"id"`` (str): Canonical capability identifier (e.g.
              ``"exact"``, ``"semantic"``, ``"structured"``, ``"combined"``,
              ``"autocomplete"``, ``"fuzzy"``).
            - ``"label"`` (str): Normalised human-readable description.
            - ``"examples"`` (list[str]): Example query strings.

            See ``_extract_search_capabilities`` for the full schema and an
            example.

    Returns:
        A 2-tuple of:

        - ``selected`` (dict[str, int]): Mapping of capability id to the
          chosen document index in *features_list*.
        - ``notes`` (list[str]): Human-readable notes about selection issues,
          such as when no compatible document was found for a capability or
          when a document had to be reused.
    """
    selected: dict[str, int] = {}
    used_indexes: set[int] = set()
    notes: list[str] = []

    for capability in capabilities:
        capability_id = str(capability.get("id", "")).strip()
        if not capability_id:
            continue
        
        # tracks the highest-scoring document index that has not yet been
        # assigned to another capability (i.e., idx not in used_indexes).
        # This is the preferred pick because it maximizes diversity:
        # each capability ideally gets its own unique sample document.
        best_unused = (-1, -1.0)

        # tracks the highest-scoring document index regardless of whether
        # it has already been used. This is the fallback: if every compatible
        # document has already been claimed by another capability,
        # the function can still reuse one rather than having no document at all.

        best_any = (-1, -1.0)
        for idx, features in enumerate(features_list):
            score = _score_doc_for_capability(features, capability_id)
            if score <= 0:
                continue
            if score > best_any[1]:
                best_any = (idx, score)
            if idx not in used_indexes and score > best_unused[1]:
                best_unused = (idx, score)

        chosen_idx = best_unused[0] if best_unused[0] >= 0 else best_any[0]
        if chosen_idx < 0:
            notes.append(f"{capability_id}: no compatible sample document found")
            continue
        if chosen_idx in used_indexes and best_unused[0] < 0:
            notes.append(f"{capability_id}: reused a document due to limited sample coverage")

        selected[capability_id] = chosen_idx
        used_indexes.add(chosen_idx)

    return selected, notes


def _trim_words(text: str, max_words: int = 8) -> str:
    words = _normalize_text(text).split()
    return " ".join(words[:max_words]).strip()


def _mutate_token_for_typo(token: str) -> str:
    if len(token) <= 2:
        return token
    pivot = len(token) // 2
    return token[:pivot] + token[pivot + 1 :]


_SEMANTIC_REWRITE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _select_semantic_source_candidate(
    semantic_candidates: list[dict[str, object]],
) -> dict[str, object]:
    best: dict[str, object] = {}
    best_score = -1.0

    for candidate in semantic_candidates:
        if not isinstance(candidate, dict):
            continue
        text = _normalize_text(candidate.get("text", ""))
        if not text:
            continue

        shape = _value_shape(text)
        token_count = int(shape.get("token_count", 0))
        tokens = [str(token).lower() for token in shape.get("tokens", [])]
        unique_token_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
        score = text_richness_score(text)
        score += min(token_count, 12) * 1.5
        score += unique_token_ratio * 8.0
        if token_count <= 2:
            score -= 15.0
        if bool(shape.get("looks_numeric", False)) or bool(shape.get("looks_date", False)):
            score -= 20.0

        if score > best_score:
            best_score = score
            best = dict(candidate)
            best["text"] = text
    return best


def _best_semantic_text_from_candidates(semantic_candidates: list[dict[str, object]]) -> str:
    selected = _select_semantic_source_candidate(semantic_candidates)
    return _normalize_text(selected.get("text", ""))


def _is_english_like(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    shape = _value_shape(normalized)
    alpha_ratio = float(shape.get("alpha_ratio", 0.0))
    if alpha_ratio < 0.55:
        return False

    alpha_chars = [ch for ch in normalized if ch.isalpha()]
    if not alpha_chars:
        return False
    ascii_alpha_ratio = sum(1 for ch in alpha_chars if ch.isascii()) / len(alpha_chars)
    return ascii_alpha_ratio >= 0.8


def _extract_concept_tokens(text: str) -> list[str]:
    shape = _value_shape(_normalize_text(text))
    concepts: list[str] = []
    seen: set[str] = set()

    for raw_token in shape.get("tokens", []):
        token = str(raw_token).strip().lower()
        if not token or len(token) < 3 or token.isdigit():
            continue
        if token in _SEMANTIC_REWRITE_STOPWORDS or token in seen:
            continue
        seen.add(token)
        concepts.append(token)
        if len(concepts) >= 6:
            break
    return concepts


def _expand_tokens_with_wordnet(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    expansions: list[str] = []
    seen: set[str] = {str(token).lower() for token in tokens}
    noun_pos = getattr(wn, "NOUN", None)
    verb_pos = getattr(wn, "VERB", None)

    def _append_candidate(raw_value: str) -> None:
        value = _normalize_text(raw_value.replace("_", " ")).lower()
        if not value or value in seen:
            return
        seen.add(value)
        expansions.append(value)

    try:
        for token in tokens[:3]:
            synsets = []
            if noun_pos is not None:
                synsets.extend(wn.synsets(token, pos=noun_pos))
            if verb_pos is not None:
                synsets.extend(wn.synsets(token, pos=verb_pos))
            if not synsets:
                continue

            first_synset = synsets[0]
            lemmas = first_synset.lemmas()
            if lemmas:
                _append_candidate(lemmas[0].name())

            hypernyms = first_synset.hypernyms()
            if hypernyms:
                hyper_lemmas = hypernyms[0].lemmas()
                if hyper_lemmas:
                    _append_candidate(hyper_lemmas[0].name())

            if len(expansions) >= 4:
                break
    except LookupError:
        return []
    except Exception:
        return []

    return expansions[:4]


def _compose_semantic_query(
    original_text: str,
    concept_tokens: list[str],
    expansions: list[str],
) -> tuple[str, bool]:
    base_tokens = concept_tokens[:6]
    if not base_tokens:
        fallback_tokens = [
            str(token).lower()
            for token in _value_shape(original_text).get("tokens", [])
            if len(str(token)) >= 3 and not str(token).isdigit()
        ]
        base_tokens = fallback_tokens[:6]

    base = _normalize_text(" ".join(base_tokens))
    if not base:
        return "", False

    if expansions:
        related = ", ".join(expansions[:2])
        return _normalize_text(f"about {base} and related ideas like {related}"), True

    base_shape = _value_shape(base)
    token_count = int(base_shape.get("token_count", 0))
    if token_count >= 4 and len(base) >= 12:
        return _normalize_text(f"about {base}"), True
    return "", False


def _rewrite_semantic_example(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    concept_tokens = _extract_concept_tokens(normalized)
    expansions: list[str] = []
    if _is_english_like(normalized):
        expansions = _expand_tokens_with_wordnet(concept_tokens)

    rewritten, confidence_high = _compose_semantic_query(
        original_text=normalized,
        concept_tokens=concept_tokens,
        expansions=expansions,
    )
    if not confidence_high:
        return normalized[:120]

    final_text = _normalize_text(rewritten)
    return final_text[:120] if final_text else normalized[:120]


def _infer_capability_examples_from_features(
    capability_id: str,
    features: dict[str, object],
) -> list[str]:
    exact_candidates = features.get("exact_candidates", [])
    semantic_candidates = features.get("semantic_candidates", [])
    structured_candidates = features.get("structured_candidates", [])
    anchor_tokens = features.get("anchor_tokens", [])
    if not isinstance(exact_candidates, list):
        exact_candidates = []
    if not isinstance(semantic_candidates, list):
        semantic_candidates = []
    if not isinstance(structured_candidates, list):
        structured_candidates = []
    if not isinstance(anchor_tokens, list):
        anchor_tokens = []

    if capability_id == "exact":
        if not exact_candidates:
            return []
        best = next(
            iter(
                sorted(
                    exact_candidates,
                    key=lambda item: (
                        0 if str(item.get("query_mode", "")) == "term" else 1,
                        -len(str(item.get("text", ""))),
                    ),
                )
            ),
            {},
        )
        text = _normalize_text(best.get("text", ""))
        return [text] if text else []

    if capability_id == "semantic":
        selected = _select_semantic_source_candidate(semantic_candidates)
        source_text = _normalize_text(selected.get("text", ""))
        if not source_text:
            return []
        rewritten = _rewrite_semantic_example(source_text)
        return [rewritten] if rewritten else [source_text]

    if capability_id == "structured":
        structured = next(iter(structured_candidates), None)
        if not isinstance(structured, dict):
            return []
        field_name = str(structured.get("field", "")).split(".")[-1]
        value_text = _normalize_text(str(structured.get("value", "")))
        if not field_name or not value_text:
            return []
        return [f"{field_name}: {value_text}"]

    if capability_id == "combined":
        selected = _select_semantic_source_candidate(semantic_candidates)
        source_text = _normalize_text(selected.get("text", ""))
        semantic_text = _rewrite_semantic_example(source_text) if source_text else ""
        structured = next(iter(structured_candidates), None)
        if not semantic_text or not isinstance(structured, dict):
            return []
        field_name = str(structured.get("field", "")).split(".")[-1]
        value_text = _normalize_text(str(structured.get("value", "")))
        if not field_name or not value_text:
            return []
        return [f"{semantic_text} with {field_name} {value_text}"]

    if capability_id == "autocomplete":
        token_text, _ = _first_anchor_token(anchor_tokens, min_len=3)
        if not token_text:
            return []
        if len(token_text) > 4:
            prefix_len = min(6, len(token_text) - 1)
            return [token_text[:prefix_len]]
        return [token_text]

    if capability_id == "fuzzy":
        token_text, _ = _first_anchor_token(anchor_tokens, min_len=5)
        if not token_text:
            return []
        return [_mutate_token_for_typo(token_text)]

    return []


def _build_suggestion_entry(capability: dict[str, object], features: dict[str, object]) -> dict[str, object] | None:
    capability_id = str(capability.get("id", "")).strip().lower()
    exact_candidates = features.get("exact_candidates", [])
    phrase_candidates = features.get("phrase_candidates", [])
    semantic_candidates = features.get("semantic_candidates", [])
    structured_candidates = features.get("structured_candidates", [])
    anchor_tokens = features.get("anchor_tokens", [])
    if not isinstance(anchor_tokens, list):
        anchor_tokens = []
    capability_examples = capability.get("examples", [])

    def _first_capability_example() -> str:
        if not isinstance(capability_examples, list):
            return ""
        for item in capability_examples:
            normalized = _normalize_text(item)
            if normalized:
                return normalized
        return ""

    def _prefer_text_field() -> str:
        """Pick a single field from the document's feature candidates for text-like search.

        Prefer fields that support phrase/semantic/full-text search over keyword-only
        fields.  Order: phrase > semantic > exact (non-.keyword) > exact (any).

        .keyword subfields are for exact match/aggregation; we prefer the parent
        text field when it exists, and only fall back to .keyword if nothing else.
        Returns the first non-empty field found, or "" if none.
        """
        # 1. phrase_candidates: match_phrase fields (usually text-type, full-text searchable)
        for candidate in phrase_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field:
                return field
        # 2. semantic_candidates: fields used for semantic/neural search (typically longer text)
        for candidate in semantic_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field:
                return field
        # 3. exact_candidates: prefer fields NOT ending with .keyword (avoid keyword subfield)
        for candidate in exact_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field and not field.endswith(".keyword"):
                return field
        # 4. exact_candidates: fallback to any field, including .keyword subfields
        for candidate in exact_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field:
                return field
        return ""

    def _prefer_keyword_field() -> str:
        for candidate in exact_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field.endswith(".keyword"):
                return field
        for candidate in exact_candidates:
            field = _normalize_text(candidate.get("field", ""))
            if field:
                return field
        return ""

    if capability_id == "exact":
        if not exact_candidates:
            return None
        best = next(
            iter(
                sorted(
                    exact_candidates,
                    key=lambda item: (
                        0 if str(item.get("query_mode", "")) == "term" else 1,
                        -len(str(item.get("text", ""))),
                    ),
                )
            ),
            {},
        )
        suggestion_text = _normalize_text(str(best.get("text", "")))
        if len(suggestion_text) < 2:
            return None
        return {
            "text": suggestion_text[:120],
            "capability": "exact",
            "query_mode": str(best.get("query_mode", "default")),
            "field": str(best.get("field", "")),
            "value": "",
            "case_insensitive": bool(best.get("case_insensitive", False)),
        }

    if capability_id == "semantic":
        planner_example = _first_capability_example()
        if planner_example:
            suggestion_text = planner_example
        else:
            best_text = _best_semantic_text_from_candidates(semantic_candidates)
            if not best_text:
                return None
            suggestion_text = _trim_words(best_text, 7)
        field = _prefer_text_field()
        query_mode = "hybrid"
        value = ""
    elif capability_id == "structured":
        structured = next(iter(structured_candidates), None)
        if structured is None:
            return None
        field_name = str(structured.get("field", "")).split(".")[-1]
        value_text = _normalize_text(str(structured.get("value", "")))
        if not field_name or not value_text:
            return None
        suggestion_text = f"{field_name}: {value_text}"
        field = _normalize_text(structured.get("field", ""))
        query_mode = "structured_filter"
        value = value_text
    elif capability_id == "combined":
        planner_example = _first_capability_example()
        structured = next(iter(structured_candidates), None)
        if structured is None:
            return None
        field_name = str(structured.get("field", "")).split(".")[-1]
        value_text = _normalize_text(str(structured.get("value", "")))
        if not field_name or not value_text:
            return None
        if planner_example and " with " in planner_example.lower():
            suggestion_text = planner_example
        else:
            semantic_text = planner_example or _best_semantic_text_from_candidates(semantic_candidates)
            if not semantic_text:
                return None
            suggestion_text = f"{_trim_words(semantic_text, 6)} with {field_name} {value_text}"
        field = _normalize_text(structured.get("field", ""))
        query_mode = "hybrid_structured"
        value = value_text
    elif capability_id == "autocomplete":
        token, token_field = _first_anchor_token(anchor_tokens, min_len=3)
        if not token:
            return None
        if len(token) > 4:
            prefix_len = min(6, len(token) - 1)
            suggestion_text = token[:prefix_len]
        else:
            suggestion_text = token
        field = token_field or _prefer_keyword_field() or _prefer_text_field()
        query_mode = "prefix"
        value = ""
    elif capability_id == "fuzzy":
        token, token_field = _first_anchor_token(anchor_tokens, min_len=5)
        if not token:
            return None
        suggestion_text = _mutate_token_for_typo(token)
        field = token_field or _prefer_text_field()
        query_mode = "fuzzy"
        value = ""
    else:
        return None

    normalized = _normalize_text(suggestion_text)
    if len(normalized) < 2:
        return None
    return {
        "text": normalized[:120],
        "capability": capability_id,
        "query_mode": query_mode,
        "field": field,
        "value": value,
        "case_insensitive": False,
    }


def _dedupe_suggestion_meta(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        text = _normalize_text(entry.get("text", ""))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        item = dict(entry)
        item["text"] = text
        deduped.append(item)
    return deduped


def _find_suggestion_meta(index_name: str, query_text: str) -> dict[str, object] | None:
    query_key = _normalized_query_key(query_text)
    if not query_key:
        return None
    for entry in _SEARCH_UI_SUGGESTION_META_BY_INDEX.get(index_name, []):
        if _normalized_query_key(entry.get("text", "")) == query_key:
            return entry
    return None


@tool
def apply_capability_driven_verification(
    worker_output: str,
    index_name: str = "",
    count: int = 10,
    id_prefix: str = "verification",
) -> dict[str, object]:
    global _LAST_VERIFICATION_INDEX

    effective_count = max(1, min(count, 100))
    target_index = (index_name or "").strip()
    result: dict[str, object] = {
        "applied": False,
        "index_name": target_index,
        "capabilities": [],
        "indexed_count": 0,
        "doc_ids": [],
        "notes": [],
    }

    if not target_index:
        result["notes"] = [
            "index_name is required for apply_capability_driven_verification; "
            "fallback index resolution is disabled."
        ]
        return result

    capabilities = _extract_search_capabilities(worker_output)
    result["capabilities"] = [str(item.get("id", "")) for item in capabilities if item.get("id")]
    if not capabilities:
        _SEARCH_UI_SUGGESTION_META_BY_INDEX.pop(target_index, None)
        result["notes"] = ["worker output has no Search Capabilities section"]
        return result

    candidate_limit = max(60, min(200, effective_count * 20))
    candidate_docs = [
        doc
        for doc in get_sample_docs_payload(candidate_limit)
        if isinstance(doc, dict)
    ]
    if not candidate_docs:
        result["notes"] = ["no sample documents available for capability-driven selection"]
        return result

    try:
        opensearch_client = _create_client()
    except Exception as e:
        result["notes"] = [f"failed to connect to OpenSearch: {e}"]
        return result

    field_specs = _extract_index_field_specs(opensearch_client, target_index)
    existing_field_types = {
        field_name: str(spec.get("type", "")).strip().lower()
        for field_name, spec in field_specs.items()
        if isinstance(spec, dict)
    }

    features_list = [_extract_doc_features(doc, field_specs) for doc in candidate_docs]
    selected_by_capability, notes = _select_docs_by_capability(features_list, capabilities)

    # Infer capability examples from selected candidate docs instead of relying on
    # planner-provided quoted examples, which may not reflect real data.
    for capability in capabilities:
        capability_id = _normalize_text(capability.get("id", "")).lower()
        if not capability_id:
            continue
        idx = selected_by_capability.get(capability_id)
        if idx is None or idx < 0 or idx >= len(features_list):
            continue
        inferred_examples = _infer_capability_examples_from_features(capability_id, features_list[idx])
        capability["examples"] = inferred_examples

    selected_indexes_for_indexing: list[int] = []
    for capability in capabilities:
        capability_id = str(capability.get("id", ""))
        idx = selected_by_capability.get(capability_id)
        if idx is None:
            continue
        if idx not in selected_indexes_for_indexing:
            selected_indexes_for_indexing.append(idx)

    for idx in range(len(features_list)):
        if len(selected_indexes_for_indexing) >= effective_count:
            break
        if idx in selected_indexes_for_indexing:
            continue
        selected_indexes_for_indexing.append(idx)

    if not selected_indexes_for_indexing:
        result["notes"] = notes + ["no documents selected for indexing"]
        return result

    existing_ids = list(_VERIFICATION_DOC_TRACKER.get(target_index, []))
    for existing_id in existing_ids:
        try:
            opensearch_client.delete(index=target_index, id=existing_id, ignore=[404])
        except Exception:
            continue

    indexed_ids: list[str] = []
    index_errors: list[str] = []
    for offset, doc_idx in enumerate(selected_indexes_for_indexing, start=1):
        doc_id = f"{id_prefix}-{offset}"
        doc_source = features_list[doc_idx]["source"]
        try:
            opensearch_client.index(index=target_index, body=doc_source, id=doc_id)
            indexed_ids.append(doc_id)
        except Exception as e:
            index_errors.append(f"{doc_id}: {e}")

    if indexed_ids:
        opensearch_client.indices.refresh(index=target_index)
        _VERIFICATION_DOC_TRACKER[target_index] = indexed_ids
        _LAST_VERIFICATION_INDEX = target_index

    suggestion_entries: list[dict[str, object]] = []
    for capability in capabilities:
        capability_id = str(capability.get("id", ""))
        idx = selected_by_capability.get(capability_id)
        if idx is None:
            continue
        entry = _build_suggestion_entry(capability, features_list[idx])
        if entry is not None:
            suggestion_entries.append(entry)

    if suggestion_entries:
        _SEARCH_UI_SUGGESTION_META_BY_INDEX[target_index] = _dedupe_suggestion_meta(suggestion_entries)
    else:
        _SEARCH_UI_SUGGESTION_META_BY_INDEX.pop(target_index, None)

    notes.extend(index_errors)
    result["notes"] = notes
    result["indexed_count"] = len(indexed_ids)
    result["doc_ids"] = indexed_ids
    result["applied"] = bool(indexed_ids)
    return result


def _suggestion_candidates_from_doc(source: dict) -> list[str]:
    """Extract search suggestion candidates from a document, ranked by value quality.

    Instead of relying on hardcoded field name hints (which don't generalize
    across unknown customer schemas), we score each scalar value by its
    characteristics: multi-word, mostly-alphabetic strings in a reasonable
    length range make the best suggestions.
    """
    if not isinstance(source, dict):
        return []

    scored: list[tuple[float, str]] = []

    for value in source.values():
        if value is None or isinstance(value, (dict, list)):
            continue
        shape = _value_shape(str(value))
        compact = str(shape["text"])
        length = int(shape["length"])

        if length < 4 or length > 80:
            continue
        if shape["looks_numeric"] or shape["looks_date"]:
            continue

        # Use shared text_richness_score for consistent scoring across modules
        score = text_richness_score(str(value))
        scored.append((score, compact))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored]

def _search_ui_suggestions(
    index_name: str, max_count: int = 6
) -> tuple[list[str], list[dict[str, object]]]:
    """Generate search suggestions for the given index.
    
    Parameters:
    - index_name: the name of the index to generate suggestions for
    - max_count: Show only a handful of top suggestions to the user.

    Returns:
    - A tuple containing a list of suggestions and a list of suggestion metadata
    """
    deduped: list[str] = []
    deduped_meta: list[dict[str, object]] = []
    seen: set[str] = set()

    def _append(text_value: object, meta_entry: dict[str, object] | None = None) -> None:
        # _normalize_text collapses whitespace; if the result is an empty string, it's skipped.
        text = _normalize_text(text_value)
        if not text:
            return
        key = text.lower()
        # The lowercased text is checked against seen. If it's already there, it's a duplicate and skipped.
        if key in seen:
            return
        seen.add(key)
        deduped.append(text)
        if meta_entry is not None:
            merged = dict(meta_entry)
            merged["text"] = text
            merged["capability"] = _normalize_text(merged.get("capability", "")).lower()
            merged["query_mode"] = _normalize_text(merged.get("query_mode", "")) or "default"
            merged["field"] = _normalize_text(merged.get("field", ""))
            merged["value"] = _normalize_text(merged.get("value", ""))
            merged["case_insensitive"] = bool(merged.get("case_insensitive", False))
        else:
            merged = {
                "text": text,
                "capability": "",
                "query_mode": "default",
                "field": "",
                "value": "",
                "case_insensitive": False,
            }
        deduped_meta.append(merged)

    for entry in _SEARCH_UI_SUGGESTION_META_BY_INDEX.get(index_name, []):
        _append(entry.get("text", ""), entry)
        if len(deduped) >= max_count:
            return deduped, deduped_meta

    if index_name:
        try:
            opensearch_client = _create_client()
            response = opensearch_client.search(
                index=index_name,
                # fetch more than needed to ensure we get enough unique suggestions.
                body={"size": max_count * 4, "query": {"match_all": {}}},
            )
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                for suggestion in _suggestion_candidates_from_doc(source):
                    _append(suggestion)
                    # early return: the function stops processing well before all max_count * 4 docs are examined. 
                    if len(deduped) >= max_count:
                        return deduped, deduped_meta
        except Exception:
            pass

    if not deduped:
        # If we still don't have enough unique suggestions, fetch some more sample documents.
        for doc in get_sample_docs_payload(max_count * 2):
            for suggestion in _suggestion_candidates_from_doc(doc):
                _append(suggestion)
                # early return: the function stops processing well before all max_count * 2 docs are examined. 
                if len(deduped) >= max_count:
                    return deduped, deduped_meta

    # If there's truly no data to draw from, returning an empty list
    # is more honest than showing fake suggestions that won't match anything.
    return deduped[:max_count], deduped_meta[:max_count]


def _search_ui_preview_text(source: dict) -> str:
    candidates = _suggestion_candidates_from_doc(source)
    if candidates:
        return candidates[0]
    if source:
        for value in source.values():
            if value is None or isinstance(value, (dict, list)):
                continue
            text = " ".join(str(value).split())
            if text:
                return text[:180]
    return "(No preview text)"


def _resolve_autocomplete_fields(
    field_specs: dict[str, dict[str, str]],
    preferred_field: str = "",
    limit: int = 4,
) -> list[str]:
    """Resolve queryable fields for autocomplete prefix matching."""
    selected: list[str] = []
    seen: set[str] = set()

    def _append(field_name: str) -> None:
        normalized = _normalize_text(field_name)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        selected.append(normalized)

    preferred = _normalize_text(preferred_field)
    if preferred:
        resolved_field, resolved_spec = _resolve_field_spec_for_doc_key(preferred, field_specs)
        candidate = resolved_field or preferred
        candidate_type = str(resolved_spec.get("type", "")).strip().lower()
        if candidate_type in {"keyword", "constant_keyword", "text"}:
            _append(candidate)

    keyword_fields = [
        field_name
        for field_name, spec in field_specs.items()
        if str(spec.get("type", "")).strip().lower() in {"keyword", "constant_keyword"}
    ]
    text_fields = [
        field_name
        for field_name, spec in field_specs.items()
        if str(spec.get("type", "")).strip().lower() == "text"
    ]

    def _rank(field_name: str) -> tuple[int, int, str]:
        return (field_name.count("."), len(field_name), field_name)

    for field_name in sorted(keyword_fields, key=_rank):
        _append(field_name)
        if len(selected) >= max(1, limit):
            return selected

    for field_name in sorted(text_fields, key=_rank):
        _append(field_name)
        if len(selected) >= max(1, limit):
            return selected

    return selected


def _extract_values_from_source_by_path(source: object, field_path: str) -> list[object]:
    """Extract scalar values from source for a dotted field path."""
    path = _normalize_text(field_path)
    if not path:
        return []
    segments = [segment for segment in path.split(".") if segment]
    if not segments:
        return []

    values: list[object] = []

    def _walk(node: object, idx: int) -> None:
        if idx >= len(segments):
            if isinstance(node, list):
                for item in node:
                    _walk(item, idx)
                return
            if isinstance(node, dict) or node is None:
                return
            values.append(node)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item, idx)
            return

        if not isinstance(node, dict):
            return

        key = segments[idx]
        if key not in node:
            return
        _walk(node.get(key), idx + 1)

    _walk(source, 0)
    return values


def _source_field_variants(field_name: str) -> list[str]:
    normalized = _normalize_text(field_name)
    if not normalized:
        return []
    variants = [normalized]
    if normalized.endswith(".keyword"):
        base_field = normalized[:-8]
        if base_field:
            variants.insert(0, base_field)
    return variants


def _search_ui_autocomplete(
    index_name: str,
    prefix_text: str,
    size: int = 8,
    preferred_field: str = "",
) -> dict[str, object]:
    """Return distinct autocomplete options for a prefix."""
    target_index = _normalize_text(index_name)
    prefix = _normalize_text(prefix_text)
    effective_size = max(1, min(size, 20))
    if not target_index or not prefix:
        return {
            "index": target_index,
            "prefix": prefix,
            "field": "",
            "options": [],
            "error": "",
        }

    try:
        opensearch_client = _create_client()
        field_specs = _extract_index_field_specs(opensearch_client, target_index)
        fields = _resolve_autocomplete_fields(
            field_specs=field_specs,
            preferred_field=preferred_field,
            limit=4,
        )
        if not fields:
            return {
                "index": target_index,
                "prefix": prefix,
                "field": "",
                "options": [],
                "error": "No suitable autocomplete fields found.",
            }

        should_clauses = [
            {"prefix": {field_name: {"value": prefix}}}
            for field_name in fields
        ]
        body = {
            "size": max(effective_size * 8, 24),
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }
        response = opensearch_client.search(index=target_index, body=body)

        options: list[str] = []
        seen: set[str] = set()
        prefix_lower = prefix.lower()

        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            if not isinstance(source, dict):
                continue

            for field_name in fields:
                for variant in _source_field_variants(field_name):
                    raw_values = _extract_values_from_source_by_path(source, variant)
                    for raw_value in raw_values:
                        candidate = _normalize_text(raw_value)
                        if not candidate:
                            continue
                        if not candidate.lower().startswith(prefix_lower):
                            continue
                        key = candidate.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        options.append(candidate[:120])
                        if len(options) >= effective_size:
                            return {
                                "index": target_index,
                                "prefix": prefix,
                                "field": fields[0],
                                "options": options,
                                "error": "",
                            }

        return {
            "index": target_index,
            "prefix": prefix,
            "field": fields[0],
            "options": options,
            "error": "",
        }
    except Exception as e:
        return {
            "index": target_index,
            "prefix": prefix,
            "field": "",
            "options": [],
            "error": str(e),
        }


_NUMERIC_FIELD_TYPES = {
    "byte",
    "short",
    "integer",
    "long",
    "float",
    "half_float",
    "double",
    "scaled_float",
}
_KEYWORD_FIELD_TYPES = {"keyword", "constant_keyword"}


def _resolve_text_query_fields(field_specs: dict[str, dict[str, str]], limit: int = 6) -> list[str]:
    """Select the best text/keyword fields from the index mapping for query targeting.

    Instead of hardcoded field-name hints (which are data-specific and don't
    generalise across unknown customer schemas), we rank fields by structural
    signals that work for any mapping:

    - Nesting depth: top-level fields (no dots) are usually primary content
      fields; deeply nested ones are less likely to be main search targets.
    - Name length: shorter names tend to be the core fields (e.g. "title")
      vs. generated/auxiliary ones (e.g. "metadata_extracted_title_v2").

    Prefers text-type fields; falls back to keyword-type if no text fields
    exist.  Returns ["*"] as a last resort to match all fields.

    Args:
        field_specs: Mapping of field path to its index mapping metadata.
        limit: Maximum number of fields to return (default 6).

    Returns:
        A list of field names, ordered by relevance, up to *limit* entries.
    """
    # Collect text-type fields (exclude .keyword sub-fields)
    text_fields = [
        field
        for field, spec in field_specs.items()
        if spec.get("type") == "text" and not field.endswith(".keyword")
    ]
    # Fallback: keyword-type fields (exclude .keyword sub-fields)
    keyword_fields = [
        field
        for field, spec in field_specs.items()
        if spec.get("type") in _KEYWORD_FIELD_TYPES and not field.endswith(".keyword")
    ]

    def _score(field_name: str) -> tuple[int, int]:
        # Primary sort: nesting depth (fewer dots = more top-level = better)
        depth = field_name.count(".")
        # Secondary sort: name length (shorter = more likely a core field)
        return depth, len(field_name)

    ranked = sorted(text_fields, key=_score)
    if not ranked:
        ranked = sorted(keyword_fields, key=_score)
    selected = ranked[: max(1, limit)]
    return selected if selected else ["*"]


def _resolve_semantic_runtime_hints(
    opensearch_client: OpenSearch,
    index_name: str,
    field_specs: dict[str, dict[str, str]],
) -> dict[str, str]:
    vector_fields = [
        field
        for field, spec in field_specs.items()
        if spec.get("type") == "knn_vector"
    ]
    vector_field = ""
    if vector_fields:
        preferred = sorted(
            vector_fields,
            key=lambda item: (
                0 if ("embedding" in item.lower() or "vector" in item.lower()) else 1,
                len(item),
                item,
            ),
        )
        vector_field = preferred[0]

    default_pipeline = ""
    model_id = ""
    source_field = ""

    try:
        settings_response = opensearch_client.indices.get_settings(index=index_name)
        index_settings = next(iter(settings_response.values()), {})
        default_pipeline = _normalize_text(
            index_settings.get("settings", {}).get("index", {}).get("default_pipeline", "")
        )
    except Exception:
        default_pipeline = ""

    if default_pipeline:
        try:
            pipeline_response = opensearch_client.ingest.get_pipeline(id=default_pipeline)
            pipeline = pipeline_response.get(default_pipeline, {})
            processors = pipeline.get("processors", [])
            for processor in processors:
                if not isinstance(processor, dict):
                    continue
                embedding = processor.get("text_embedding")
                if not isinstance(embedding, dict):
                    continue
                candidate_model = _normalize_text(embedding.get("model_id", ""))
                field_map = embedding.get("field_map", {})
                if not candidate_model:
                    continue
                if isinstance(field_map, dict) and field_map:
                    if vector_field:
                        for source, target in field_map.items():
                            if _normalize_text(target) == vector_field:
                                model_id = candidate_model
                                source_field = _normalize_text(source)
                                break
                        if model_id:
                            break
                    if not model_id:
                        first_source, first_target = next(iter(field_map.items()))
                        model_id = candidate_model
                        source_field = _normalize_text(first_source)
                        if not vector_field:
                            vector_field = _normalize_text(first_target)
                else:
                    model_id = candidate_model
                    break
        except Exception:
            pass

    return {
        "vector_field": vector_field,
        "model_id": model_id,
        "default_pipeline": default_pipeline,
        "source_field": source_field,
    }


def _build_default_lexical_query(query: str, fields: list[str]) -> dict:
    body: dict[str, object] = {
        "query": query,
        "fields": fields or ["*"],
    }
    if any(field != "*" for field in fields):
        body["fuzziness"] = "AUTO"
    return {"multi_match": body}


def _build_default_lexical_body(query: str, size: int, fields: list[str]) -> dict:
    return {
        "size": size,
        "query": _build_default_lexical_query(query=query, fields=fields),
    }


def _coerce_structured_value(raw_value: str, field_type: str) -> object:
    normalized = _normalize_text(raw_value)
    lowered = normalized.lower()
    if field_type in _NUMERIC_FIELD_TYPES:
        if field_type in {"byte", "short", "integer", "long"}:
            try:
                return int(float(normalized))
            except Exception:
                return normalized
        try:
            return float(normalized)
        except Exception:
            return normalized

    if field_type == "boolean":
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return normalized


def _parse_structured_clause(
    query_text: str,
    suggestion_meta: dict[str, object] | None,
    field_specs: dict[str, dict[str, str]],
) -> tuple[dict | None, str]:
    field_name = _normalize_text((suggestion_meta or {}).get("field", ""))
    value_text = _normalize_text((suggestion_meta or {}).get("value", ""))

    if not field_name or not value_text:
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*(.+)\s*$", query_text)
        if match:
            field_name = _normalize_text(match.group(1))
            value_text = _normalize_text(match.group(2))

    if not field_name or not value_text:
        return None, "structured query missing field/value"

    resolved_field, resolved_spec = _resolve_field_spec_for_doc_key(field_name, field_specs)
    target_field = resolved_field or field_name
    field_type = str(resolved_spec.get("type", "")).strip()

    if field_type == "text":
        return {"match_phrase": {target_field: value_text}}, ""

    coerced_value = _coerce_structured_value(value_text, field_type)
    return {"term": {target_field: {"value": coerced_value}}}, ""


def _build_neural_clause(query: str, vector_field: str, model_id: str, size: int) -> dict:
    return {
        "neural": {
            vector_field: {
                "query_text": query,
                "model_id": model_id,
                "k": max(size, 10),
            }
        }
    }


def _search_ui_search(
    index_name: str,
    query_text: str,
    size: int = 10,
    debug: bool = False,
) -> dict:
    if not index_name:
        return {
            "error": "Missing index name.",
            "hits": [],
            "took_ms": 0,
            "query_mode": "",
            "capability": "",
            "used_semantic": False,
            "fallback_reason": "",
        }

    opensearch_client = _create_client()
    query = query_text.strip()
    capability = "manual" if query else ""
    query_mode = "match_all"
    used_semantic = False
    fallback_reason = ""
    executed_body: dict[str, object] = {"size": size, "query": {"match_all": {}}}

    field_specs = _extract_index_field_specs(opensearch_client, index_name)
    lexical_fields = _resolve_text_query_fields(field_specs)
    suggestion_meta = _find_suggestion_meta(index_name, query) if query else None
    if suggestion_meta is not None:
        resolved_capability = _normalize_text(suggestion_meta.get("capability", "")).lower()
        if resolved_capability:
            capability = resolved_capability

    if query:
        runtime_hints = _resolve_semantic_runtime_hints(opensearch_client, index_name, field_specs)
        vector_field = runtime_hints.get("vector_field", "")
        model_id = runtime_hints.get("model_id", "")
        semantic_ready = bool(vector_field and model_id)
        lexical_query = _build_default_lexical_query(query=query, fields=lexical_fields)

        if capability == "exact" and suggestion_meta is not None:
            exact_mode = _normalize_text(suggestion_meta.get("query_mode", ""))
            field = _normalize_text(suggestion_meta.get("field", ""))
            query_value = query.lower() if bool(suggestion_meta.get("case_insensitive", False)) else query
            if exact_mode == "term" and field:
                executed_body = {
                    "size": size,
                    "query": {
                        "term": {
                            field: {
                                "value": query_value,
                            }
                        }
                    },
                }
                query_mode = "exact_term"
            elif exact_mode == "match_phrase" and field:
                executed_body = {
                    "size": size,
                    "query": {
                        "match_phrase": {
                            field: query,
                        }
                    },
                }
                query_mode = "exact_match_phrase"
            else:
                executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
                query_mode = "exact_bm25_fallback"
        elif capability == "structured":
            structured_clause, structured_error = _parse_structured_clause(
                query_text=query,
                suggestion_meta=suggestion_meta,
                field_specs=field_specs,
            )
            if structured_clause is None:
                fallback_reason = structured_error
                executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
                query_mode = "structured_bm25_fallback"
            else:
                if "match_phrase" in structured_clause:
                    executed_body = {
                        "size": size,
                        "query": structured_clause,
                    }
                else:
                    executed_body = {
                        "size": size,
                        "query": {
                            "bool": {
                                "filter": [structured_clause],
                                "must": [{"match_all": {}}],
                            }
                        },
                    }
                query_mode = "structured_filter"
        elif capability == "autocomplete":
            field = _normalize_text((suggestion_meta or {}).get("field", ""))
            if field:
                executed_body = {
                    "size": size,
                    "query": {
                        "prefix": {
                            field: {
                                "value": query,
                            }
                        }
                    },
                }
                query_mode = "autocomplete_prefix"
            else:
                executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
                query_mode = "autocomplete_bm25_fallback"
                fallback_reason = "autocomplete field unresolved"
        elif capability == "fuzzy":
            field = _normalize_text((suggestion_meta or {}).get("field", ""))
            if field:
                executed_body = {
                    "size": size,
                    "query": {
                        "match": {
                            field: {
                                "query": query,
                                "fuzziness": "AUTO",
                            }
                        }
                    },
                }
                query_mode = "fuzzy_match"
            else:
                executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
                query_mode = "fuzzy_bm25_fallback"
                fallback_reason = "fuzzy field unresolved"
        elif capability in {"semantic", "combined", "manual"}:
            structured_clause = None
            if capability == "combined":
                structured_clause, structured_error = _parse_structured_clause(
                    query_text=query,
                    suggestion_meta=suggestion_meta,
                    field_specs=field_specs,
                )
                if structured_clause is None:
                    fallback_reason = structured_error

            if semantic_ready:
                neural_query = _build_neural_clause(
                    query=query,
                    vector_field=vector_field,
                    model_id=model_id,
                    size=size,
                )
                base_hybrid = {
                    "hybrid": {
                        "queries": [
                            lexical_query,
                            neural_query,
                        ]
                    }
                }
                if capability == "combined" and structured_clause is not None:
                    if "match_phrase" in structured_clause:
                        executed_body = {
                            "size": size,
                            "query": {
                                "bool": {
                                    "must": [
                                        base_hybrid,
                                        structured_clause,
                                    ]
                                }
                            },
                        }
                    else:
                        executed_body = {
                            "size": size,
                            "query": {
                                "bool": {
                                    "must": [base_hybrid],
                                    "filter": [structured_clause],
                                }
                            },
                        }
                    query_mode = "combined_hybrid"
                elif capability == "semantic":
                    executed_body = {
                        "size": size,
                        "query": base_hybrid,
                    }
                    query_mode = "semantic_hybrid"
                else:
                    executed_body = {
                        "size": size,
                        "query": base_hybrid,
                    }
                    query_mode = "hybrid_default"
                used_semantic = True
            else:
                missing_parts: list[str] = []
                if not vector_field:
                    missing_parts.append("vector_field")
                if not model_id:
                    missing_parts.append("model_id")
                missing_text = ", ".join(missing_parts) if missing_parts else "semantic runtime unavailable"
                fallback_reason = (
                    f"{fallback_reason}; semantic runtime unresolved ({missing_text})"
                    if fallback_reason
                    else f"semantic runtime unresolved ({missing_text})"
                )
                executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
                query_mode = f"{capability}_bm25_fallback"
        else:
            executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
            query_mode = "bm25_default"
    else:
        executed_body = {"size": size, "query": {"match_all": {}}}

    try:
        response = opensearch_client.search(index=index_name, body=executed_body)
    except Exception as query_error:
        if query:
            fallback_reason = (
                f"{fallback_reason}; primary query failed: {query_error}"
                if fallback_reason
                else f"primary query failed: {query_error}"
            )
            executed_body = _build_default_lexical_body(query=query, size=size, fields=lexical_fields)
            response = opensearch_client.search(index=index_name, body=executed_body)
            used_semantic = False
            query_mode = f"{query_mode}_fallback_bm25"
        else:
            raise

    hits_out: list[dict] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        hits_out.append(
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "preview": _search_ui_preview_text(source),
                "source": source,
            }
        )
    return {
        "error": "",
        "hits": hits_out,
        "total": response.get("hits", {}).get("total", {}).get("value", len(hits_out)),
        "took_ms": response.get("took", 0),
        "query_mode": query_mode,
        "capability": capability,
        "used_semantic": used_semantic,
        "fallback_reason": fallback_reason,
        **({"query_body": executed_body} if debug else {}),
    }


def _resolve_default_index(preferred_index: str = "") -> str:
    if preferred_index:
        return preferred_index
    if _LAST_VERIFICATION_INDEX:
        return _LAST_VERIFICATION_INDEX
    try:
        opensearch_client = _create_client()
        indices = opensearch_client.cat.indices(format="json")
        names = [
            item.get("index", "")
            for item in indices
            if item.get("index") and not item.get("index", "").startswith(".")
        ]
        if names:
            return names[0]
    except Exception:
        pass
    return ""


def _search_ui_public_url() -> str:
    public_host = "localhost" if SEARCH_UI_HOST in {"0.0.0.0", "::"} else SEARCH_UI_HOST
    return f"http://{public_host}:{SEARCH_UI_PORT}"


def _resolve_search_ui_asset(path: str) -> Path | None:
    normalized = path.strip()
    if normalized in {"", "/"}:
        normalized = "/index.html"

    relative_path = normalized.lstrip("/")
    candidate = (SEARCH_UI_STATIC_DIR / relative_path).resolve()
    root = SEARCH_UI_STATIC_DIR.resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _search_ui_content_type(path: Path) -> str:
    return _SEARCH_UI_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class _SearchUIRequestHandler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_file(self, path: Path, status: int = 200) -> None:
        payload = path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", _search_ui_content_type(path))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/health":
            self._write_json({"ok": True, "default_index": _SEARCH_UI_DEFAULT_INDEX})
            return

        if parsed.path == "/api/config":
            self._write_json({"default_index": _SEARCH_UI_DEFAULT_INDEX})
            return

        if parsed.path == "/api/suggestions":
            index_name = params.get("index", [""])[0] or _SEARCH_UI_DEFAULT_INDEX
            suggestions, suggestion_meta = _search_ui_suggestions(index_name, max_count=6)
            self._write_json(
                {
                    "suggestions": suggestions,
                    "suggestion_meta": suggestion_meta,
                    "index": index_name,
                }
            )
            return

        if parsed.path == "/api/autocomplete":
            index_name = params.get("index", [""])[0] or _SEARCH_UI_DEFAULT_INDEX
            prefix_text = params.get("q", [""])[0]
            field_name = params.get("field", [""])[0]
            try:
                size = int(params.get("size", ["8"])[0])
            except ValueError:
                size = 8
            size = max(1, min(size, 20))
            result = _search_ui_autocomplete(
                index_name=index_name,
                prefix_text=prefix_text,
                size=size,
                preferred_field=field_name,
            )
            self._write_json(result)
            return

        if parsed.path == "/api/search":
            index_name = params.get("index", [""])[0] or _SEARCH_UI_DEFAULT_INDEX
            query_text = params.get("q", [""])[0]
            debug_param = params.get("debug", ["0"])[0].strip().lower()
            debug_mode = debug_param in {"1", "true", "yes", "on"}
            try:
                size = int(params.get("size", ["20"])[0])
            except ValueError:
                size = 20
            size = max(1, min(size, 50))

            try:
                result = _search_ui_search(
                    index_name=index_name,
                    query_text=query_text,
                    size=size,
                    debug=debug_mode,
                )
                self._write_json(result)
            except Exception as e:
                self._write_json(
                    {
                        "error": str(e),
                        "hits": [],
                        "took_ms": 0,
                        "query_mode": "",
                        "capability": "",
                        "used_semantic": False,
                        "fallback_reason": "",
                    },
                    status=500,
                )
            return

        static_asset = _resolve_search_ui_asset(parsed.path)
        if static_asset is not None:
            self._write_file(static_asset)
            return

        self._write_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def _start_search_ui_server(preferred_index: str = "") -> str:
    global _SEARCH_UI_SERVER, _SEARCH_UI_SERVER_THREAD, _SEARCH_UI_DEFAULT_INDEX

    with _SEARCH_UI_SERVER_LOCK:
        if not SEARCH_UI_STATIC_DIR.exists():
            raise RuntimeError(f"Search UI static directory not found: {SEARCH_UI_STATIC_DIR}")
        if _resolve_search_ui_asset("/index.html") is None:
            raise RuntimeError("Search UI entry file missing: index.html")

        resolved_index = _resolve_default_index(preferred_index)
        if resolved_index:
            _SEARCH_UI_DEFAULT_INDEX = resolved_index

        if _SEARCH_UI_SERVER is not None and _SEARCH_UI_SERVER_THREAD is not None:
            if _SEARCH_UI_SERVER_THREAD.is_alive():
                return _search_ui_public_url()

        server = ThreadingHTTPServer((SEARCH_UI_HOST, SEARCH_UI_PORT), _SearchUIRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _SEARCH_UI_SERVER = server
        _SEARCH_UI_SERVER_THREAD = thread

    return _search_ui_public_url()

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

def set_ml_settings(opensearch_client: OpenSearch | None = None) -> None:
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
def create_index(index_name: str, body: dict = None, replace_if_exists: bool = True) -> str:
    """Create an OpenSearch index with the specified configuration.

    Args:
        index_name: The name of the index to create.
        body: The configuration body for the index (settings and mappings).
        replace_if_exists: Delete and recreate the index if it already exists.

    Returns:
        str: Success message or error.
    """
    if body is None:
        body = {}
    if not isinstance(body, dict):
        body = {}
    knn_engine_updates = _normalize_knn_method_engines(body)

    sample_docs = [
        doc
        for doc in get_sample_docs_payload(limit=200)
        if isinstance(doc, dict)
    ]
    requested_field_types = _extract_declared_field_types_from_index_body(body)

    def _index_exists(opensearch_client: OpenSearch, target_index: str) -> bool:
        try:
            exists = opensearch_client.indices.exists(index=target_index)
            if isinstance(exists, bool):
                return exists
            return bool(exists)
        except Exception:
            try:
                mapping = opensearch_client.indices.get_mapping(index=target_index)
                return isinstance(mapping, dict) and bool(mapping)
            except Exception:
                return False

    try:
        opensearch_client = _create_client()
    except Exception as e:
        return f"Failed to create index '{index_name}': {e}"

    existed_before = _index_exists(opensearch_client, index_name)
    if existed_before and replace_if_exists:
        try:
            opensearch_client.indices.delete(index=index_name, ignore=[404])
            _VERIFICATION_DOC_TRACKER.pop(index_name, None)
            _SEARCH_UI_SUGGESTION_META_BY_INDEX.pop(index_name, None)
        except Exception as e:
            return f"Failed to recreate index '{index_name}': failed to delete existing index: {e}"

    if existed_before and not replace_if_exists:
        existing_field_specs = _extract_index_field_specs(opensearch_client, index_name)
        existing_field_types = {
            field_name: str(spec.get("type", "")).strip().lower()
            for field_name, spec in existing_field_specs.items()
            if isinstance(spec, dict)
        }

        mapping_mismatches = _collect_requested_vs_existing_field_type_mismatches(
            requested_field_types=requested_field_types,
            existing_field_types=existing_field_types,
        )
        if mapping_mismatches:
            return (
                f"Error: Index '{index_name}' already exists with mappings incompatible with the requested schema. "
                "Delete and recreate the index, or use replace_if_exists=true. "
                + " ".join(mapping_mismatches)
            )
        return f"Index '{index_name}' already exists."

    try:
        opensearch_client.indices.create(index=index_name, body=body)
        normalized_note = ""
        if knn_engine_updates:
            normalized_note = (
                " Normalized k-NN method engine settings: "
                + "; ".join(knn_engine_updates)
                + "."
            )
        if existed_before and replace_if_exists:
            return f"Index '{index_name}' recreated successfully.{normalized_note}"
        return f"Index '{index_name}' created successfully.{normalized_note}"
    except Exception as e:
        error = str(e)
        lowered = error.lower()
        if "resource_already_exists_exception" in lowered or "already exists" in lowered:
            if replace_if_exists:
                try:
                    opensearch_client.indices.delete(index=index_name, ignore=[404])
                    opensearch_client.indices.create(index=index_name, body=body)
                    _VERIFICATION_DOC_TRACKER.pop(index_name, None)
                    _SEARCH_UI_SUGGESTION_META_BY_INDEX.pop(index_name, None)
                    normalized_note = ""
                    if knn_engine_updates:
                        normalized_note = (
                            " Normalized k-NN method engine settings: "
                            + "; ".join(knn_engine_updates)
                            + "."
                        )
                    return f"Index '{index_name}' recreated successfully.{normalized_note}"
                except Exception as recreate_error:
                    return f"Failed to recreate index '{index_name}': {recreate_error}"

            existing_field_specs = _extract_index_field_specs(opensearch_client, index_name)
            existing_field_types = {
                field_name: str(spec.get("type", "")).strip().lower()
                for field_name, spec in existing_field_specs.items()
                if isinstance(spec, dict)
            }

            mapping_mismatches = _collect_requested_vs_existing_field_type_mismatches(
                requested_field_types=requested_field_types,
                existing_field_types=existing_field_types,
            )
            if mapping_mismatches:
                return (
                    f"Error: Index '{index_name}' already exists with mappings incompatible with the requested schema. "
                    "Delete and recreate the index, or use replace_if_exists=true. "
                    + " ".join(mapping_mismatches)
                )
            return f"Index '{index_name}' already exists."
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
def create_and_attach_pipeline(
    pipeline_name: str,
    pipeline_body: dict,
    index_name: str,
    pipeline_type: str = "ingest",
    replace_if_exists: bool = True,
    is_hybrid_search: bool = False,
    hybrid_weights: list[float] | None = None,
) -> str:
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
        replace_if_exists: Delete and recreate pipeline when it already exists.
        is_hybrid_search: Whether the search pipeline is for hybrid lexical+semantic score blending.
        hybrid_weights: Weight array in [lexical, semantic] order.

    Returns:
        str: Success message or error.
    """
    def _extract_pipeline_source_fields(body: dict) -> list[str]:
        if not isinstance(body, dict):
            return []

        processors = body.get("processors")
        if not isinstance(processors, list):
            return []

        source_fields: list[str] = []
        seen: set[str] = set()
        for processor in processors:
            if not isinstance(processor, dict):
                continue
            for processor_config in processor.values():
                if not isinstance(processor_config, dict):
                    continue
                field_map = processor_config.get("field_map")
                if not isinstance(field_map, dict):
                    continue
                for source_field in field_map.keys():
                    name = str(source_field).strip().lower()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    source_fields.append(name)
        return source_fields

    def _extract_mapped_fields(opensearch_client: OpenSearch, target_index: str) -> dict[str, str]:
        mapped_fields: dict[str, str] = {}
        response = opensearch_client.indices.get_mapping(index=target_index)
        if not isinstance(response, dict) or not response:
            return mapped_fields

        selected_mapping = response.get(target_index)
        if not isinstance(selected_mapping, dict):
            selected_mapping = next(iter(response.values()), {})
        mappings = selected_mapping.get("mappings", {}) if isinstance(selected_mapping, dict) else {}
        properties = mappings.get("properties", {}) if isinstance(mappings, dict) else {}

        def _walk(props: dict, prefix: str = "") -> None:
            for field_name, config in props.items():
                if not isinstance(config, dict):
                    continue
                full_name = f"{prefix}.{field_name}" if prefix else field_name
                field_type = config.get("type")
                if isinstance(field_type, str):
                    mapped_fields[full_name] = field_type
                nested_props = config.get("properties")
                if isinstance(nested_props, dict):
                    _walk(nested_props, full_name)

        if isinstance(properties, dict):
            _walk(properties)
        return mapped_fields

    def _choose_best_source_field(requested_field: str, mapped_fields: dict[str, str]) -> str:
        # Strict matching only: exact match, case-insensitive full path, then case-insensitive leaf-name.
        # No heuristic fallback to arbitrary ranked candidates.
        requested = requested_field.strip()
        if not requested:
            return ""
        if requested in mapped_fields:
            return requested

        requested_lower = requested.lower()
        for candidate in mapped_fields.keys():
            if candidate.lower() == requested_lower:
                return candidate

        leaf_matches = [
            candidate for candidate in mapped_fields.keys()
            if candidate.split(".")[-1].lower() == requested_lower
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        return ""

    def _normalize_ingest_pipeline_body(body: dict, mapped_fields: dict[str, str]) -> tuple[dict, list[str], list[str]]:
        if not isinstance(body, dict):
            return body, [], []

        normalized = json.loads(json.dumps(body))
        processors = normalized.get("processors")
        if not isinstance(processors, list):
            return normalized, [], []

        remap_notes: list[str] = []
        unresolved: list[str] = []
        for processor in processors:
            if not isinstance(processor, dict):
                continue

            for processor_name, processor_config in processor.items():
                if not isinstance(processor_config, dict):
                    continue
                field_map = processor_config.get("field_map")
                if not isinstance(field_map, dict) or not field_map:
                    continue

                rewritten_map: dict[str, str] = {}
                for source_field, target_field in field_map.items():
                    source_text = str(source_field).strip()
                    chosen_source = _choose_best_source_field(source_text, mapped_fields)
                    if not chosen_source:
                        unresolved.append(f"{processor_name}: '{source_text}'")
                        continue
                    rewritten_map[chosen_source] = target_field
                    if chosen_source.lower() != source_text.lower():
                        remap_notes.append(
                            f"{processor_name}: '{source_text}' -> '{chosen_source}'"
                        )

                processor_config["field_map"] = rewritten_map

        return normalized, remap_notes, unresolved

    def _resolve_hybrid_weights(weights: list[float] | None) -> tuple[list[float], str]:
        default_weights = [0.5, 0.5]
        if weights is None:
            return default_weights, ""
        if not isinstance(weights, list) or len(weights) != 2:
            return default_weights, (
                "hybrid_weights must be a list with exactly two numeric values "
                "in [lexical, semantic] order."
            )
        try:
            lexical = float(weights[0])
            semantic = float(weights[1])
        except Exception:
            return default_weights, (
                "hybrid_weights must be numeric in [lexical, semantic] order."
            )
        if lexical < 0 or semantic < 0:
            return default_weights, "hybrid_weights values must be non-negative."
        total = lexical + semantic
        if total <= 0:
            return default_weights, "hybrid_weights sum must be greater than zero."
        return [lexical / total, semantic / total], ""

    def _build_default_hybrid_search_pipeline_body(weights: list[float]) -> dict:
        return {
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": weights},
                        },
                    }
                }
            ]
        }

    def _find_normalization_processor(body: dict) -> dict[str, object] | None:
        processors = body.get("phase_results_processors")
        if not isinstance(processors, list):
            return None
        for processor in processors:
            if not isinstance(processor, dict):
                continue
            normalization = processor.get("normalization-processor")
            if isinstance(normalization, dict):
                return normalization
        return None

    def _normalize_hybrid_search_pipeline_body(body: dict, weights: list[float]) -> tuple[dict, str]:
        if not isinstance(body, dict) or not body:
            return _build_default_hybrid_search_pipeline_body(weights), ""

        normalized = json.loads(json.dumps(body))
        normalization_processor = _find_normalization_processor(normalized)
        if normalization_processor is None:
            return {}, (
                "Hybrid search pipeline must include 'phase_results_processors' "
                "with a 'normalization-processor' entry."
            )

        normalization = normalization_processor.get("normalization")
        if not isinstance(normalization, dict):
            normalization = {}
        technique = _normalize_text(normalization.get("technique", ""))
        if not technique:
            normalization["technique"] = "min_max"
        normalization_processor["normalization"] = normalization

        combination = normalization_processor.get("combination")
        if not isinstance(combination, dict):
            combination = {}
        combination_technique = _normalize_text(combination.get("technique", ""))
        if not combination_technique:
            combination["technique"] = "arithmetic_mean"
        parameters = combination.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        parameters["weights"] = weights
        combination["parameters"] = parameters
        normalization_processor["combination"] = combination
        return normalized, ""

    try:
        opensearch_client = _create_client()
        normalized_pipeline_body = pipeline_body
        remap_notes: list[str] = []
        existed_before = False

        if pipeline_type == "ingest":
            mapped_fields = _extract_mapped_fields(opensearch_client, index_name)
            normalized_pipeline_body, remap_notes, unresolved = _normalize_ingest_pipeline_body(
                pipeline_body,
                mapped_fields,
            )
            if unresolved:
                requested_fields = _extract_pipeline_source_fields(pipeline_body)
                available_fields = sorted(mapped_fields.keys())
                set_ingest_source_field_hints([])
                return (
                    "Error: Ingest pipeline field_map source fields are invalid for this index mapping. "
                    f"Requested source fields: {requested_fields or ['(none)']}. "
                    f"Unresolved mappings: {unresolved}. "
                    f"Available mapped fields: {available_fields or ['(none)']}. "
                    "Please rerun planning/execution and update the pipeline field_map to use existing source fields."
                )

            try:
                existing = opensearch_client.ingest.get_pipeline(id=pipeline_name)
                existed_before = isinstance(existing, dict) and bool(existing)
            except Exception:
                existed_before = False

            if existed_before and replace_if_exists:
                try:
                    opensearch_client.ingest.delete_pipeline(id=pipeline_name)
                except Exception as e:
                    return f"Failed to recreate ingest pipeline '{pipeline_name}': {e}"

            if existed_before and not replace_if_exists:
                settings = {"index.default_pipeline": pipeline_name}
                opensearch_client.indices.put_settings(index=index_name, body=settings)
                set_ingest_source_field_hints(_extract_pipeline_source_fields(normalized_pipeline_body))
                return (
                    f"Ingest pipeline '{pipeline_name}' already exists and is attached to index "
                    f"'{index_name}'."
                )

            opensearch_client.ingest.put_pipeline(id=pipeline_name, body=normalized_pipeline_body)
            settings = {"index.default_pipeline": pipeline_name}
        elif pipeline_type == "search":
            resolved_weights = [0.5, 0.5]
            normalized_search_pipeline_body = pipeline_body
            if is_hybrid_search:
                resolved_weights, weights_error = _resolve_hybrid_weights(hybrid_weights)
                if weights_error:
                    return f"Error: Invalid hybrid search weights: {weights_error}"
                normalized_search_pipeline_body, hybrid_error = _normalize_hybrid_search_pipeline_body(
                    pipeline_body,
                    resolved_weights,
                )
                if hybrid_error:
                    return f"Error: {hybrid_error}"

            try:
                opensearch_client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_name}")
                existed_before = True
            except Exception:
                existed_before = False

            if existed_before and replace_if_exists:
                try:
                    opensearch_client.transport.perform_request("DELETE", f"/_search/pipeline/{pipeline_name}")
                except Exception as e:
                    return f"Failed to recreate search pipeline '{pipeline_name}': {e}"

            if existed_before and not replace_if_exists:
                settings = {"index.search.default_pipeline": pipeline_name}
                opensearch_client.indices.put_settings(index=index_name, body=settings)
                return (
                    f"Search pipeline '{pipeline_name}' already exists and is attached to index "
                    f"'{index_name}'."
                )

            # Use low-level client for search pipeline to ensure compatibility
            opensearch_client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{pipeline_name}",
                body=normalized_search_pipeline_body,
            )
            settings = {"index.search.default_pipeline": pipeline_name}
        else:
            return f"Error: Invalid pipeline_type '{pipeline_type}'. Must be 'ingest' or 'search'."

        # Always re-attach after create/recreate so index settings are guaranteed.
        opensearch_client.indices.put_settings(index=index_name, body=settings)
        action = "recreated" if (existed_before and replace_if_exists) else "created"
        if pipeline_type == "ingest":
            set_ingest_source_field_hints(_extract_pipeline_source_fields(normalized_pipeline_body))
            if remap_notes:
                return (
                    f"{pipeline_type.capitalize()} pipeline '{pipeline_name}' {action} and attached to index '{index_name}' successfully. "
                    f"field remap: {'; '.join(remap_notes)}"
                )
        return (
            f"{pipeline_type.capitalize()} pipeline '{pipeline_name}' {action} and attached to index "
            f"'{index_name}' successfully."
        )

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
def index_verification_docs(index_name: str, count: int = 10, id_prefix: str = "verification") -> str:
    """Index verification docs from collected sample data for UI testing.

    Args:
        index_name: Target index name.
        count: Number of docs to index (default 10, max 100).
        id_prefix: Prefix for generated doc IDs.

    Returns:
        str: JSON summary of indexed IDs and any errors.
    """
    global _LAST_VERIFICATION_INDEX

    effective_count = max(1, min(count, 100))
    docs = get_sample_docs_payload(effective_count)
    if not docs:
        return (
            "Failed to index verification docs: no sample docs available. "
            "Collect sample data first."
        )

    opensearch_client = _create_client()
    indexed_ids: list[str] = []
    errors: list[str] = []

    for i, doc in enumerate(docs, start=1):
        doc_id = f"{id_prefix}-{i}"
        try:
            opensearch_client.index(index=index_name, body=doc, id=doc_id)
            indexed_ids.append(doc_id)
        except Exception as e:
            errors.append(f"{doc_id}: {e}")

    if indexed_ids:
        opensearch_client.indices.refresh(index=index_name)

    existing = set(_VERIFICATION_DOC_TRACKER.get(index_name, []))
    if index_name not in _VERIFICATION_DOC_TRACKER:
        _VERIFICATION_DOC_TRACKER[index_name] = []
    for doc_id in indexed_ids:
        if doc_id not in existing:
            _VERIFICATION_DOC_TRACKER[index_name].append(doc_id)

    if indexed_ids:
        _LAST_VERIFICATION_INDEX = index_name

    result = {
        "index_name": index_name,
        "requested_count": effective_count,
        "indexed_count": len(indexed_ids),
        "doc_ids": indexed_ids,
        "errors": errors,
        "cleanup_hint": "Verification docs were kept. Run cleanup_verification_docs when user asks.",
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def launch_search_ui(index_name: str = "") -> str:
    """Launch local React Search Builder UI for interactive testing."""
    try:
        url = _start_search_ui_server(index_name)
        selected_index = _resolve_default_index(index_name)
    except Exception as e:
        return f"Failed to launch Search Builder UI: {e}"

    if selected_index:
        return (
            f"Search Builder UI is running at: {url}\n"
            f"Default index: {selected_index}\n"
            "You can run queries immediately and inspect returned documents."
        )
    return (
        f"Search Builder UI is running at: {url}\n"
        "No default index selected. Enter an index name in the UI to start searching."
    )


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


@tool
def cleanup_verification_docs(index_name: str = "") -> str:
    """Delete tracked verification docs (only when user explicitly requests cleanup)."""
    targets = [index_name] if index_name else list(_VERIFICATION_DOC_TRACKER.keys())
    if not targets:
        return "No tracked verification docs found to clean up."

    opensearch_client = _create_client()
    deleted_count = 0
    errors: list[str] = []

    for target in targets:
        doc_ids = list(_VERIFICATION_DOC_TRACKER.get(target, []))
        if not doc_ids:
            continue

        remaining_ids: list[str] = []
        for doc_id in doc_ids:
            try:
                opensearch_client.delete(index=target, id=doc_id, ignore=[404])
                deleted_count += 1
            except Exception as e:
                errors.append(f"{target}/{doc_id}: {e}")
                remaining_ids.append(doc_id)

        if remaining_ids:
            _VERIFICATION_DOC_TRACKER[target] = remaining_ids
        elif target in _VERIFICATION_DOC_TRACKER:
            del _VERIFICATION_DOC_TRACKER[target]

    summary = {"deleted_docs": deleted_count, "errors": errors}
    return json.dumps(summary, ensure_ascii=False)
