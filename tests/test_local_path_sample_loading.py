import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tools import (
    _extract_path_candidate,
    clear_sample_doc,
    get_sample_doc,
    submit_sample_doc_from_local_file,
)


@pytest.fixture(autouse=True)
def _clear_sample_doc_state():
    clear_sample_doc()
    yield
    clear_sample_doc()


def test_extract_path_candidate_strips_sentence_trailing_period(tmp_path, monkeypatch):
    home = tmp_path / "home"
    downloads = home / "Downloads"
    downloads.mkdir(parents=True)
    data_file = downloads / "title.basics.tsv"
    data_file.write_text("id\ttitle\n1\tTest\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    message = "Data source is ~/Downloads/title.basics.tsv. Please index it."
    detected = _extract_path_candidate(message)

    assert detected == str(data_file)


def test_extract_path_candidate_supports_quoted_absolute_path_with_spaces(tmp_path):
    data_file = tmp_path / "title basics.tsv"
    data_file.write_text("id\ttitle\n1\tTest\n", encoding="utf-8")

    message = f'Use "{data_file}" for indexing.'
    detected = _extract_path_candidate(message)

    assert detected == str(data_file)


def test_submit_sample_doc_from_local_file_accepts_sentence_style_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    downloads = home / "Downloads"
    downloads.mkdir(parents=True)
    data_file = downloads / "title.basics.tsv"
    data_file.write_text(
        "tconst\tprimaryTitle\tisAdult\n"
        "tt0000001\tCarmencita\t0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    message = (
        "Help setup OpenSearch. Data for search is ~/Downloads/title.basics.tsv. "
        "Please proceed."
    )
    result = submit_sample_doc_from_local_file(message)

    assert result.startswith("Sample document loaded from")
    stored = json.loads(get_sample_doc())
    assert stored["tconst"] == "tt0000001"
    assert stored["primaryTitle"] == "Carmencita"
    assert stored["isAdult"] == "0"


def test_extract_path_candidate_returns_sanitized_missing_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    message = "Input file: ~/Downloads/missing.tsv."
    detected = _extract_path_candidate(message)

    assert detected.endswith("missing.tsv")
