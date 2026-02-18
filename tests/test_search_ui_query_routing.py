from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.opensearch_ops_tools as tools


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, index, body):
        self.calls.append({"index": index, "body": body})
        return {
            "hits": {
                "hits": [],
                "total": {"value": 0},
            },
            "took": 3,
        }


class FakeAutocompleteClient:
    def __init__(self, hits):
        self.calls = []
        self._hits = hits

    def search(self, index, body):
        self.calls.append({"index": index, "body": body})
        return {
            "hits": {
                "hits": self._hits,
                "total": {"value": len(self._hits)},
            },
            "took": 2,
        }



def _base_field_specs():
    return {
        "primaryTitle": {"type": "text", "normalizer": ""},
        "primaryTitle.keyword": {"type": "keyword", "normalizer": ""},
        "embedding_vector": {"type": "knn_vector", "normalizer": ""},
        "genres": {"type": "keyword", "normalizer": ""},
        "startYear": {"type": "integer", "normalizer": ""},
    }



def test_suggestion_api_returns_meta_tuple(monkeypatch):
    index_name = "imdb_movies"
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX[index_name] = [
        {
            "text": "romantic comedies from the 90s",
            "capability": "semantic",
            "query_mode": "hybrid",
            "field": "primaryTitle",
            "value": "",
            "case_insensitive": False,
        }
    ]

    class _NoClient:
        pass

    monkeypatch.setattr(tools, "_create_client", lambda: _NoClient())

    suggestions, suggestion_meta = tools._search_ui_suggestions(index_name=index_name, max_count=6)

    assert suggestions
    assert suggestions[0] == "romantic comedies from the 90s"
    assert suggestion_meta
    assert suggestion_meta[0]["capability"] == "semantic"



def test_semantic_chip_routes_to_hybrid(monkeypatch):
    client = FakeClient()
    index_name = "imdb_movies"
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX[index_name] = [
        {
            "text": "romantic paris films",
            "capability": "semantic",
            "query_mode": "hybrid",
            "field": "primaryTitle",
            "value": "",
            "case_insensitive": False,
        }
    ]

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(tools, "_extract_index_field_specs", lambda *_: _base_field_specs())
    monkeypatch.setattr(
        tools,
        "_resolve_semantic_runtime_hints",
        lambda *_: {
            "vector_field": "embedding_vector",
            "model_id": "model-123",
            "default_pipeline": "imdb_hybrid_pipeline",
            "source_field": "combined_text",
        },
    )

    result = tools._search_ui_search(
        index_name=index_name,
        query_text="romantic paris films",
        size=10,
        debug=True,
    )

    assert result["error"] == ""
    assert result["query_mode"] == "semantic_hybrid"
    assert result["used_semantic"] is True
    assert result["capability"] == "semantic"

    assert client.calls
    query = client.calls[-1]["body"]["query"]
    assert "hybrid" in query
    queries = query["hybrid"]["queries"]
    assert any("neural" in item for item in queries)



def test_manual_query_defaults_to_hybrid(monkeypatch):
    client = FakeClient()
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX.clear()

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(tools, "_extract_index_field_specs", lambda *_: _base_field_specs())
    monkeypatch.setattr(
        tools,
        "_resolve_semantic_runtime_hints",
        lambda *_: {
            "vector_field": "embedding_vector",
            "model_id": "model-456",
            "default_pipeline": "imdb_hybrid_pipeline",
            "source_field": "combined_text",
        },
    )

    result = tools._search_ui_search(
        index_name="imdb_movies",
        query_text="space exploration documentaries",
        size=10,
        debug=True,
    )

    assert result["query_mode"] == "hybrid_default"
    assert result["capability"] == "manual"
    assert result["used_semantic"] is True
    assert "hybrid" in result["query_body"]["query"]



def test_exact_chip_keeps_exact_term_query(monkeypatch):
    client = FakeClient()
    index_name = "imdb_movies"
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX[index_name] = [
        {
            "text": "tt000013",
            "capability": "exact",
            "query_mode": "term",
            "field": "tconst.keyword",
            "value": "",
            "case_insensitive": False,
        }
    ]

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(tools, "_extract_index_field_specs", lambda *_: _base_field_specs())
    monkeypatch.setattr(
        tools,
        "_resolve_semantic_runtime_hints",
        lambda *_: {
            "vector_field": "embedding_vector",
            "model_id": "model-xyz",
            "default_pipeline": "imdb_hybrid_pipeline",
            "source_field": "combined_text",
        },
    )

    result = tools._search_ui_search(index_name=index_name, query_text="tt000013", size=10, debug=True)

    assert result["query_mode"] == "exact_term"
    assert result["used_semantic"] is False
    query = result["query_body"]["query"]
    assert "term" in query
    assert query["term"]["tconst.keyword"]["value"] == "tt000013"



def test_semantic_falls_back_to_bm25_when_runtime_missing(monkeypatch):
    client = FakeClient()
    index_name = "imdb_movies"
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX[index_name] = [
        {
            "text": "romantic paris films",
            "capability": "semantic",
            "query_mode": "hybrid",
            "field": "primaryTitle",
            "value": "",
            "case_insensitive": False,
        }
    ]

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(tools, "_extract_index_field_specs", lambda *_: _base_field_specs())
    monkeypatch.setattr(
        tools,
        "_resolve_semantic_runtime_hints",
        lambda *_: {
            "vector_field": "embedding_vector",
            "model_id": "",
            "default_pipeline": "imdb_hybrid_pipeline",
            "source_field": "combined_text",
        },
    )

    result = tools._search_ui_search(
        index_name=index_name,
        query_text="romantic paris films",
        size=10,
        debug=True,
    )

    assert result["query_mode"] == "semantic_bm25_fallback"
    assert result["used_semantic"] is False
    assert "semantic runtime unresolved" in result["fallback_reason"]
    assert "multi_match" in result["query_body"]["query"]


def test_autocomplete_returns_prefix_options_for_keyword_field(monkeypatch):
    client = FakeAutocompleteClient(
        [
            {"_source": {"tconst": "tt0000001", "primaryTitle": "Carmencita"}},
            {"_source": {"tconst": "tt0012345", "primaryTitle": "Le clown et ses chiens"}},
            {"_source": {"tconst": "nm9999999", "primaryTitle": "Not matching prefix"}},
        ]
    )

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(
        tools,
        "_extract_index_field_specs",
        lambda *_: {
            "tconst": {"type": "keyword", "normalizer": ""},
            "primaryTitle": {"type": "text", "normalizer": ""},
        },
    )

    result = tools._search_ui_autocomplete(
        index_name="imdb_movies",
        prefix_text="tt",
        size=5,
        preferred_field="tconst",
    )

    assert result["error"] == ""
    assert result["field"] == "tconst"
    assert result["options"] == ["tt0000001", "tt0012345"]
    assert client.calls
    should = client.calls[-1]["body"]["query"]["bool"]["should"]
    assert any("prefix" in clause and "tconst" in clause["prefix"] for clause in should)


def test_autocomplete_reads_source_value_for_keyword_subfield(monkeypatch):
    client = FakeAutocompleteClient(
        [
            {"_source": {"tconst": "tt0000100"}},
            {"_source": {"tconst": "tt0000200"}},
        ]
    )

    monkeypatch.setattr(tools, "_create_client", lambda: client)
    monkeypatch.setattr(
        tools,
        "_extract_index_field_specs",
        lambda *_: {
            "tconst.keyword": {"type": "keyword", "normalizer": ""},
        },
    )

    result = tools._search_ui_autocomplete(
        index_name="imdb_movies",
        prefix_text="tt0",
        size=5,
        preferred_field="tconst.keyword",
    )

    assert result["error"] == ""
    assert result["options"] == ["tt0000100", "tt0000200"]
