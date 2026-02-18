import json
import os
import re
import csv
from typing import Any, Dict, Optional
from pathlib import Path
from html import unescape
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from strands import tool
from scripts.shared import text_richness_score

_SAMPLE_DOC: Optional[Dict[str, Any]] = None
_SAMPLE_SOURCE_LOCAL_FILE: Optional[str] = None
_INGEST_SOURCE_FIELD_HINTS: tuple[str, ...] = ()


def clear_sample_doc() -> None:
    """Clear the stored sample document (used on workflow reset)."""
    global _SAMPLE_DOC, _SAMPLE_SOURCE_LOCAL_FILE, _INGEST_SOURCE_FIELD_HINTS
    _SAMPLE_DOC = None
    _SAMPLE_SOURCE_LOCAL_FILE = None
    _INGEST_SOURCE_FIELD_HINTS = ()


def set_ingest_source_field_hints(fields: list[str] | tuple[str, ...] | None) -> None:
    """Store source field hints extracted from ingest pipeline field_map."""
    global _INGEST_SOURCE_FIELD_HINTS

    if not fields:
        _INGEST_SOURCE_FIELD_HINTS = ("text",)
        return

    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        name = str(field).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)

    _INGEST_SOURCE_FIELD_HINTS = tuple(normalized) if normalized else ()


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
    global _SAMPLE_DOC, _SAMPLE_SOURCE_LOCAL_FILE

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
    _SAMPLE_SOURCE_LOCAL_FILE = None
    return "Sample document stored."


def _expand_local_path(path_text: str) -> Path:
    return Path(os.path.expandvars(path_text)).expanduser()


def _strip_trailing_path_punctuation(path_text: str) -> str:
    # Sentence punctuation often follows pasted paths in free-form prompts.
    return path_text.rstrip(").,;!?")


def _extract_path_candidate(path_or_text: str) -> str:
    text = path_or_text.strip()
    if not text:
        return ""

    candidates: list[str] = []

    raw = text.strip('"').strip("'")
    if raw and (
        raw.startswith("~/")
        or raw.startswith("/")
        or raw.startswith("./")
        or raw.startswith("../")
        or re.fullmatch(
            r"[^\s,;]+\.(tsv|tab|csv|jsonl|ndjson|json|txt)",
            raw,
            flags=re.IGNORECASE,
        )
    ):
        candidates.append(raw)

    # Quoted path support, including spaces in file names.
    for match in re.finditer(r"(['\"])([^'\"]+)\1", text):
        quoted = match.group(2).strip()
        if quoted.startswith("~/") or quoted.startswith("/"):
            candidates.append(quoted)

    # Unquoted inline path token inside free-form text.
    for match in re.finditer(r"(~\/[^\s,;]+|\/[^\s,;]+)", text):
        candidates.append(match.group(1))
    for match in re.finditer(
        r"((?:\./|\../)?[^\s,;]+\.(?:tsv|tab|csv|jsonl|ndjson|json|txt))",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(match.group(1))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip().strip('"').strip("'")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)

    if not deduped:
        return ""

    fallback_token = _strip_trailing_path_punctuation(deduped[0]) or deduped[0]
    for candidate in deduped:
        variants = [candidate]
        stripped = _strip_trailing_path_punctuation(candidate)
        if stripped and stripped != candidate:
            variants.append(stripped)

        for variant in variants:
            resolved = _expand_local_path(variant)
            if resolved.exists():
                return str(resolved)
    return str(_expand_local_path(fallback_token))


def _extract_url_candidate(url_or_text: str) -> str:
    raw = url_or_text.strip().strip('"').strip("'")
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip(").,;")

    match = re.search(r"https?://[^\s]+", url_or_text)
    if not match:
        return ""
    return match.group(0).rstrip(").,;")


def _normalize_cell_value(value: str) -> Any:
    text = value.strip()
    if text in {"\\N", "NULL", "null", ""}:
        return None
    return text


_TABULAR_EXTENSIONS = {".tsv", ".tab", ".csv"}
_LINE_BASED_EXTENSIONS = _TABULAR_EXTENSIONS | {".jsonl", ".ndjson", ".txt"}
_SUPPORTED_LOCAL_DATA_EXTENSIONS = _LINE_BASED_EXTENSIONS | {".json"}
_SCRIPT_LABELS = {
    "latin": "Latin",
    "cyrillic": "Cyrillic",
    "cjk": "CJK",
    "japanese": "Japanese",
    "korean": "Korean",
    "arabic": "Arabic",
    "hebrew": "Hebrew",
    "devanagari": "Devanagari",
    "other_alpha": "Other alphabetic",
}


def _pick_directory_sample_file(directory: Path, limit: int = 500) -> tuple[Optional[Path], int]:
    def _extract_candidate_fields(candidate_path: Path) -> set[str]:
        extension = candidate_path.suffix.lower()

        try:
            if extension in _TABULAR_EXTENSIONS:
                with candidate_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        header_line = line.rstrip("\r\n")
                        if not header_line:
                            continue
                        delimiter = "\t" if extension in {".tsv", ".tab"} else ","
                        header = next(csv.reader([header_line], delimiter=delimiter))
                        return {
                            col.strip().lower()
                            for col in header
                            if col and col.strip()
                        }

            if extension in {".jsonl", ".ndjson"}:
                with candidate_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        raw = line.strip()
                        if not raw:
                            continue
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            return {
                                str(key).strip().lower()
                                for key in parsed.keys()
                                if str(key).strip()
                            }
                        return set()
        except Exception:
            return set()

        return set()

    candidates: list[Path] = []
    try:
        for candidate in directory.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_LOCAL_DATA_EXTENSIONS:
                candidates.append(candidate)
                if len(candidates) >= limit:
                    break
    except Exception:
        pass

    if not candidates:
        return None, 0

    # most machine-parseable and schema-rich first
    extension_priority = {
        ".tsv": 0,
        ".tab": 1,
        ".csv": 2,
        ".jsonl": 3,
        ".ndjson": 4,
        ".json": 5,
        ".txt": 6,
    }

    preferred_fields = set(_INGEST_SOURCE_FIELD_HINTS)
    field_match_score: dict[Path, int] = {}
    for candidate in candidates:
        candidate_fields = _extract_candidate_fields(candidate)
        if not candidate_fields:
            field_match_score[candidate] = 0
            continue
        field_match_score[candidate] = len(preferred_fields & candidate_fields)

    # Candidate files are ranked by:
    # 1. Number of matched fields with pipeline source fields (higher first).
    # 2. Existing extension priority (.tsv, .tab, .csv, ...).
    # 3. Path name.
    candidates.sort(
        key=lambda p: (
            -field_match_score.get(p, 0),
            extension_priority.get(p.suffix.lower(), 99),
            str(p).lower(),
        )
    )
    return candidates[0], len(candidates)


def _count_lines_exact(file_path: Path) -> int:
    newline_count = 0
    total_bytes = 0
    last_byte = b""
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if total_bytes == 0:
        return 0
    if newline_count == 0:
        return 1
    if last_byte != b"\n":
        newline_count += 1
    return newline_count


def _estimate_line_count(file_path: Path, sample_lines: int = 4000) -> int:
    try:
        total_size = file_path.stat().st_size
    except Exception:
        return 0

    if total_size == 0:
        return 0

    sampled_bytes = 0
    sampled_lines = 0
    try:
        with file_path.open("rb") as handle:
            for _ in range(sample_lines):
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                sampled_bytes += len(line)
                sampled_lines += 1
    except Exception:
        return 0

    if sampled_lines == 0 or sampled_bytes == 0:
        return 0

    average_line_bytes = sampled_bytes / sampled_lines
    estimated_lines = int(total_size / max(average_line_bytes, 1.0))
    return max(sampled_lines, estimated_lines)


def _estimate_record_count(file_path: Path) -> tuple[Optional[int], str]:
    extension = file_path.suffix.lower()
    if extension not in _LINE_BASED_EXTENSIONS:
        return None, "unknown"

    try:
        file_size = file_path.stat().st_size
    except Exception:
        return None, "unknown"

    if file_size <= 150 * 1024 * 1024:
        line_count = _count_lines_exact(file_path)
        method = "exact"
    else:
        line_count = _estimate_line_count(file_path)
        method = "estimated"

    header_rows = 1 if extension in _TABULAR_EXTENSIONS and line_count > 0 else 0
    return max(line_count - header_rows, 0), method




def _script_bucket(char: str) -> Optional[str]:
    code = ord(char)
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return "latin"
    if 0x00C0 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF:
        return "latin"
    if 0x0400 <= code <= 0x052F:
        return "cyrillic"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "cjk"
    if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
        return "japanese"
    if 0xAC00 <= code <= 0xD7AF:
        return "korean"
    if 0x0600 <= code <= 0x06FF:
        return "arabic"
    if 0x0590 <= code <= 0x05FF:
        return "hebrew"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if char.isalpha():
        return "other_alpha"
    return None


def _infer_language_hint(parsed_doc: Dict[str, Any]) -> str:
    """Infer the likely language/script from a sample document.

    Instead of relying on hardcoded field name hints (which don't generalise
    across unknown customer schemas), we score each value by its text richness
    (alphabetic ratio, word count, length) and use the highest-scoring values
    for script detection.
    """
    # Collect all scalar string values with their text richness scores
    scored_texts: list[tuple[float, str]] = []
    for value in parsed_doc.values():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        score = text_richness_score(text)
        if score > 0:
            scored_texts.append((score, text))

    # Sort by richness descending — most alphabetic, multi-word text first
    scored_texts.sort(key=lambda item: item[0], reverse=True)
    samples = [text for _, text in scored_texts]
    if not samples:
        return "insufficient text in sample row to infer language."

    script_counts: Dict[str, int] = {}
    for text in samples[:20]:
        for char in text:
            script = _script_bucket(char)
            if script is None:
                continue
            script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return "insufficient alphabetic signal in sample row to infer language."

    total = sum(script_counts.values())
    latin_ratio = script_counts.get("latin", 0) / total
    significant_non_latin = [
        _SCRIPT_LABELS.get(name, name)
        for name, count in script_counts.items()
        if name != "latin" and (count / total) >= 0.08
    ]

    if latin_ratio >= 0.9 and not significant_non_latin:
        return "sample appears predominantly Latin script (often English-like)."
    if latin_ratio >= 0.6:
        if significant_non_latin:
            return (
                "sample appears mostly Latin with additional scripts: "
                f"{', '.join(significant_non_latin[:3])}."
            )
        return "sample appears mostly Latin script."
    top_scripts = sorted(script_counts.items(), key=lambda item: item[1], reverse=True)
    top_labels = [_SCRIPT_LABELS.get(name, name) for name, _ in top_scripts[:3]]
    return f"sample appears non-Latin or multilingual (top scripts: {', '.join(top_labels)})."


def _load_sample_record_from_file(file_path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        lines = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\r\n")
                if stripped == "":
                    continue
                lines.append(stripped)
                if len(lines) >= 2:
                    break
    except Exception as e:
        return None, f"Error reading file '{file_path}': {e}"

    if not lines:
        return None, f"Error: file '{file_path}' is empty."

    header_line = lines[0]
    sample_line = lines[1] if len(lines) > 1 else None

    extension = file_path.suffix.lower()
    delimiter = None
    if extension in {".tsv", ".tab"}:
        delimiter = "\t"
    elif extension == ".csv":
        delimiter = ","
    elif "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","

    parsed_doc: Dict[str, Any]
    if delimiter and sample_line is not None:
        header = next(csv.reader([header_line], delimiter=delimiter))
        sample_row = next(csv.reader([sample_line], delimiter=delimiter))

        if len(sample_row) < len(header):
            sample_row.extend([""] * (len(header) - len(sample_row)))
        if len(sample_row) > len(header):
            sample_row = sample_row[: len(header)]

        parsed_doc = {}
        for idx, key in enumerate(header):
            normalized_key = key.strip() or f"field_{idx + 1}"
            parsed_doc[normalized_key] = _normalize_cell_value(sample_row[idx])
    else:
        parsed_doc = {"content": sample_line or header_line}

    return parsed_doc, None


def _load_records_from_local_file(file_path: Path, limit: int = 10) -> tuple[list[Dict[str, Any]], Optional[str]]:
    effective_limit = max(1, min(limit, 200))

    try:
        lines: list[str] = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\r\n")
                if stripped == "":
                    continue
                lines.append(stripped)
                if len(lines) >= effective_limit + 1:
                    break
    except Exception as e:
        return [], f"Error reading file '{file_path}': {e}"

    if not lines:
        return [], f"Error: file '{file_path}' is empty."

    header_line = lines[0]
    extension = file_path.suffix.lower()
    delimiter = None
    if extension in {".tsv", ".tab"}:
        delimiter = "\t"
    elif extension == ".csv":
        delimiter = ","
    elif "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","

    docs: list[Dict[str, Any]] = []
    if delimiter and len(lines) > 1:
        header = next(csv.reader([header_line], delimiter=delimiter))
        for row_line in lines[1:]:
            sample_row = next(csv.reader([row_line], delimiter=delimiter))
            if len(sample_row) < len(header):
                sample_row.extend([""] * (len(header) - len(sample_row)))
            if len(sample_row) > len(header):
                sample_row = sample_row[: len(header)]

            parsed_doc: Dict[str, Any] = {}
            for idx, key in enumerate(header):
                normalized_key = key.strip() or f"field_{idx + 1}"
                parsed_doc[normalized_key] = _normalize_cell_value(sample_row[idx])
            docs.append(parsed_doc)
            if len(docs) >= effective_limit:
                break
    else:
        for line in lines[:effective_limit]:
            docs.append({"content": line})

    if not docs:
        return [], f"Error: no records found in file '{file_path}'."
    return docs, None


def get_sample_docs_payload(limit: int = 10) -> list[Dict[str, Any]]:
    """Return sample docs for verification from local source when available."""
    effective_limit = max(1, min(limit, 200))

    if _SAMPLE_SOURCE_LOCAL_FILE:
        file_path = Path(_SAMPLE_SOURCE_LOCAL_FILE)
        if file_path.exists() and file_path.is_file():
            docs, _ = _load_records_from_local_file(file_path, effective_limit)
            if docs:
                return docs[:effective_limit]

    if _SAMPLE_DOC is None:
        return []
    return [dict(_SAMPLE_DOC)]


@tool
def get_sample_docs_for_verification(limit: int = 10) -> str:
    """Return sample docs for verification indexing as JSON array."""
    return json.dumps(get_sample_docs_payload(limit), ensure_ascii=False)


@tool
def submit_sample_doc_from_local_file(path_or_text: str) -> str:
    """Load one sample record from a local file or directory and store it as sample doc.

    Args:
        path_or_text: Local file/directory path, or text containing a path like
            ~/Downloads/data.tsv.

    Returns:
        str: Status message indicating success or reason for failure.
    """
    global _SAMPLE_DOC, _SAMPLE_SOURCE_LOCAL_FILE

    resolved_text = _extract_path_candidate(path_or_text)
    if not resolved_text:
        return "Error: could not detect a local file path."

    source_path = Path(resolved_text)
    if not source_path.exists():
        return f"Error: local path not found: {source_path}"

    directory_note = ""
    if source_path.is_dir():
        selected_file, file_count = _pick_directory_sample_file(source_path)
        if selected_file is None:
            return (
                f"Error: no supported data files found under directory '{source_path}'. "
                "Supported extensions: .tsv, .tab, .csv, .jsonl, .ndjson, .json, .txt"
            )
        file_path = selected_file
        directory_note = (
            f" Source directory: '{source_path}' (detected {file_count} supported data file(s)); "
            f"using '{file_path}' as the sample file."
        )
    elif source_path.is_file():
        file_path = source_path
    else:
        return f"Error: local path is neither a file nor a directory: {source_path}"

    parsed_doc, parse_error = _load_sample_record_from_file(file_path)
    if parse_error is not None:
        return parse_error
    if parsed_doc is None:
        return f"Error: failed to parse sample record from '{file_path}'."

    _SAMPLE_DOC = parsed_doc
    _SAMPLE_SOURCE_LOCAL_FILE = str(file_path)
    field_preview = ", ".join(list(parsed_doc.keys())[:8])
    record_count, count_method = _estimate_record_count(file_path)
    if record_count is None:
        count_note = "record count could not be inferred from this file type."
    else:
        count_prefix = "inferred records (excluding header)" if file_path.suffix.lower() in _TABULAR_EXTENSIONS else "inferred records"
        method_suffix = "exact" if count_method == "exact" else "estimated"
        count_note = f"{count_prefix}: {record_count:,} ({method_suffix})."
    language_note = _infer_language_hint(parsed_doc)
    scope_note = (
        "Clarification hint: confirm whether this source is the full corpus or only a sample for future files,"
        " and whether additional languages are expected."
    )
    return (
        f"Sample document loaded from '{file_path}'. "
        f"Detected fields: {field_preview}.{directory_note} "
        f"Data profile: {count_note} Language hint: {language_note} {scope_note}"
    )


@tool
def submit_sample_doc_from_url(url_or_text: str) -> str:
    """Download one sample record from a URL and store it as sample doc.

    Args:
        url_or_text: HTTP/HTTPS URL, or text containing a URL.

    Returns:
        str: Status message indicating success or reason for failure.
    """
    global _SAMPLE_DOC, _SAMPLE_SOURCE_LOCAL_FILE

    url = _extract_url_candidate(url_or_text)
    if not url:
        return "Error: could not detect an HTTP/HTTPS URL."

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return f"Error: unsupported URL scheme '{parsed_url.scheme}'."

    try:
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenSearchAgent/1.0)"},
        )
        with urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            content_type = (response.headers.get("Content-Type", "") or "").lower()
            raw = response.read(1024 * 1024)  # Read first 1MB, enough for header + sample rows.
    except Exception as e:
        return f"Error downloading URL '{url}': {e}"

    if not raw:
        return f"Error: downloaded content from '{url}' is empty."

    text = raw.decode(charset, errors="replace")
    if not text.strip():
        return f"Error: downloaded content from '{url}' is blank."

    # JSON payload.
    if "application/json" in content_type or parsed_url.path.lower().endswith(".json"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                parsed_doc = payload
            elif isinstance(payload, list) and payload:
                first = payload[0]
                parsed_doc = first if isinstance(first, dict) else {"content": str(first)}
            else:
                parsed_doc = {"content": text.splitlines()[0]}
            _SAMPLE_DOC = parsed_doc
            _SAMPLE_SOURCE_LOCAL_FILE = None
            field_preview = ", ".join(list(parsed_doc.keys())[:8])
            return (
                f"Sample document loaded from URL '{url}'. "
                f"Detected fields: {field_preview}"
            )
        except json.JSONDecodeError:
            # Fall through to line-based parsing (some URLs return JSONL/text with incorrect headers).
            pass

    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines:
        return f"Error: no usable lines found in downloaded content from '{url}'."

    # JSONL support.
    first_line = lines[0].strip()
    if first_line.startswith("{"):
        try:
            parsed_doc = json.loads(first_line)
            if isinstance(parsed_doc, dict):
                _SAMPLE_DOC = parsed_doc
                _SAMPLE_SOURCE_LOCAL_FILE = None
                field_preview = ", ".join(list(parsed_doc.keys())[:8])
                return (
                    f"Sample document loaded from URL '{url}'. "
                    f"Detected fields: {field_preview}"
                )
        except json.JSONDecodeError:
            pass

    header_line = lines[0]
    sample_line = lines[1] if len(lines) > 1 else None
    extension = Path(parsed_url.path).suffix.lower()

    delimiter = None
    if extension in {".tsv", ".tab"}:
        delimiter = "\t"
    elif extension == ".csv":
        delimiter = ","
    elif "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","

    parsed_doc: Dict[str, Any]
    if delimiter and sample_line is not None:
        header = next(csv.reader([header_line], delimiter=delimiter))
        sample_row = next(csv.reader([sample_line], delimiter=delimiter))

        if len(sample_row) < len(header):
            sample_row.extend([""] * (len(header) - len(sample_row)))
        if len(sample_row) > len(header):
            sample_row = sample_row[: len(header)]

        parsed_doc = {}
        for idx, key in enumerate(header):
            normalized_key = key.strip() or f"field_{idx + 1}"
            parsed_doc[normalized_key] = _normalize_cell_value(sample_row[idx])
    else:
        parsed_doc = {"content": sample_line or header_line}

    _SAMPLE_DOC = parsed_doc
    _SAMPLE_SOURCE_LOCAL_FILE = None
    field_preview = ", ".join(list(parsed_doc.keys())[:8])
    return (
        f"Sample document loaded from URL '{url}'. "
        f"Detected fields: {field_preview}"
    )

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
