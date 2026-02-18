from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import worker


def test_extract_hybrid_weight_profile_parses_supported_values():
    context = """
    - Retrieval Method: Hybrid Search (Dense Vector + BM25)
    - Hybrid Weight Profile: semantic-heavy
    """

    profile = worker._extract_hybrid_weight_profile(context)

    assert profile == "semantic-heavy"


def test_resolve_hybrid_weights_semantic_heavy_lexical_semantic_scope():
    context = """
    - Retrieval Method: Hybrid Search (Dense Vector + BM25)
    - Hybrid Weight Profile: semantic-heavy
    """

    enabled, weights, profile = worker._resolve_hybrid_search_pipeline_weights(context)

    assert enabled is True
    assert weights == [0.2, 0.8]
    assert profile == "semantic-heavy"


def test_resolve_hybrid_weights_lexical_heavy_lexical_semantic_scope():
    context = """
    - Retrieval Method: Hybrid Search (Dense Vector + BM25)
    - Hybrid Weight Profile: lexical-heavy
    """

    enabled, weights, profile = worker._resolve_hybrid_search_pipeline_weights(context)

    assert enabled is True
    assert weights == [0.8, 0.2]
    assert profile == "lexical-heavy"


def test_resolve_hybrid_weights_defaults_to_balanced_when_profile_missing():
    context = """
    - Retrieval Method: Hybrid Search (Dense Vector + BM25)
    - Score Normalization: Min-Max or L2
    """

    enabled, weights, profile = worker._resolve_hybrid_search_pipeline_weights(context)

    assert enabled is True
    assert weights == [0.5, 0.5]
    assert profile == "balanced"


def test_resolve_hybrid_weights_not_enabled_for_dense_sparse_only_hybrid():
    context = """
    - Retrieval Method: Hybrid Search (Dense Vector + Sparse Vector)
    - No BM25 component
    """

    enabled, weights, profile = worker._resolve_hybrid_search_pipeline_weights(context)

    assert enabled is False
    assert weights == [0.5, 0.5]
    assert profile == ""


def test_resolve_hybrid_weights_not_enabled_for_non_hybrid():
    context = """
    - Retrieval Method: Dense Vector
    - Model: amazon.titan-embed-text-v2
    """

    enabled, weights, profile = worker._resolve_hybrid_search_pipeline_weights(context)

    assert enabled is False
    assert weights == [0.5, 0.5]
    assert profile == ""
