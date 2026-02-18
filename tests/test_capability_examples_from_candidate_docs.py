from pathlib import Path
import sys
import types
import builtins

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.opensearch_ops_tools as tools


class _FakeIndices:
    def __init__(self, mapping_response):
        self._mapping_response = mapping_response
        self.refreshed = []

    def get_mapping(self, index):
        return self._mapping_response

    def refresh(self, index):
        self.refreshed.append(index)


class _FakeClient:
    def __init__(self, mapping_response):
        self.indices = _FakeIndices(mapping_response)
        self.indexed = []
        self.deleted = []

    def index(self, index, body, id):
        self.indexed.append((index, id, body))

    def delete(self, index, id, ignore=None):
        self.deleted.append((index, id, tuple(ignore or [])))


def _mapping_response():
    return {
        "imdb_titles": {
            "mappings": {
                "properties": {
                    "primaryTitle": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "genre": {"type": "keyword"},
                    "startYear": {"type": "integer"},
                }
            }
        }
    }


def test_extract_search_capabilities_does_not_store_quoted_examples():
    worker_output = (
        "## Search Capabilities\n"
        '- Semantic: "romantic comedies in paris"\n'
    )
    capabilities = tools._extract_search_capabilities(worker_output)

    assert capabilities
    assert capabilities[0]["id"] == "semantic"
    assert capabilities[0]["examples"] == []


def test_infer_capability_examples_from_features_semantic(monkeypatch):
    features = {
        "exact_candidates": [],
        "semantic_candidates": [
            {"text": "short text", "field": "primaryTitle"},
            {"text": "space exploration mission documentary", "field": "primaryTitle"},
        ],
        "structured_candidates": [],
        "anchor_tokens": [],
    }

    # Keep deterministic behavior in tests: force lexical expansion to be unavailable.
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: [])

    examples = tools._infer_capability_examples_from_features("semantic", features)

    assert examples == ["about space exploration mission documentary"]


def test_rewrite_semantic_example_with_mocked_wordnet_expansion(monkeypatch):
    monkeypatch.setattr(tools, "_is_english_like", lambda _text: True)
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: ["metalwork", "hammer"])

    rewritten = tools._rewrite_semantic_example("Blacksmith Scene")

    assert rewritten.startswith("about blacksmith scene")
    assert "related ideas like metalwork, hammer" in rewritten


def test_rewrite_semantic_example_keeps_original_when_low_confidence_without_wordnet(monkeypatch):
    monkeypatch.setattr(tools, "_is_english_like", lambda _text: True)
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: [])

    rewritten = tools._rewrite_semantic_example("Blacksmith Scene")

    assert rewritten == "Blacksmith Scene"


def test_rewrite_semantic_example_descriptive_phrase_without_wordnet(monkeypatch):
    monkeypatch.setattr(tools, "_is_english_like", lambda _text: True)
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: [])

    rewritten = tools._rewrite_semantic_example("space exploration mission documentary")

    assert rewritten == "about space exploration mission documentary"


def test_rewrite_semantic_example_non_english_fallback_safe(monkeypatch):
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: ["should not be used"])

    source = "宇宙探査についての短編映画"
    rewritten = tools._rewrite_semantic_example(source)

    assert rewritten == source


def test_expand_tokens_with_wordnet_handles_import_error(monkeypatch):
    real_import = builtins.__import__

    def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("nltk"):
            raise ImportError("nltk unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    assert tools._expand_tokens_with_wordnet(["blacksmith"]) == []


def test_expand_tokens_with_wordnet_handles_lookup_error(monkeypatch):
    fake_wordnet = types.SimpleNamespace(
        NOUN="n",
        VERB="v",
        synsets=lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError("missing corpus")),
    )
    fake_corpus = types.ModuleType("nltk.corpus")
    fake_corpus.wordnet = fake_wordnet
    fake_nltk = types.ModuleType("nltk")
    fake_nltk.corpus = fake_corpus

    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", fake_corpus)

    assert tools._expand_tokens_with_wordnet(["blacksmith"]) == []


def test_infer_capability_examples_from_features_combined(monkeypatch):
    monkeypatch.setattr(tools, "_is_english_like", lambda _text: True)
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: [])
    features = {
        "exact_candidates": [],
        "semantic_candidates": [
            {"text": "space exploration mission documentary", "field": "description"},
        ],
        "structured_candidates": [
            {"field": "genre", "value": "Documentary", "type": "keyword"},
        ],
        "anchor_tokens": [],
    }

    examples = tools._infer_capability_examples_from_features("combined", features)

    assert examples == ["about space exploration mission documentary with genre Documentary"]


def test_apply_verification_infers_semantic_example_from_candidate_docs(monkeypatch):
    tools._SEARCH_UI_SUGGESTION_META_BY_INDEX.clear()
    tools._VERIFICATION_DOC_TRACKER.clear()

    fake_client = _FakeClient(mapping_response=_mapping_response())
    monkeypatch.setattr(tools, "_create_client", lambda: fake_client)
    monkeypatch.setattr(tools, "_expand_tokens_with_wordnet", lambda _tokens: [])
    monkeypatch.setattr(
        tools,
        "get_sample_docs_payload",
        lambda limit=200: [
            {
                "primaryTitle": "space exploration mission documentary",
                "genre": "Documentary",
                "startYear": 1969,
            }
        ],
    )

    worker_output = (
        "## Search Capabilities\n"
        '- Semantic: "romantic comedies in paris"\n'
    )

    result = tools.apply_capability_driven_verification(
        worker_output=worker_output,
        index_name="imdb_titles",
        count=1,
    )

    assert result["applied"] is True
    suggestion_meta = tools._SEARCH_UI_SUGGESTION_META_BY_INDEX.get("imdb_titles", [])
    semantic_entries = [entry for entry in suggestion_meta if entry.get("capability") == "semantic"]
    assert semantic_entries
    assert semantic_entries[0]["text"] == "about space exploration mission documentary"
    assert "romantic comedies in paris" not in semantic_entries[0]["text"].lower()
