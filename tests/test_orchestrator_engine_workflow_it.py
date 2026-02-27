import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opensearch_orchestrator.orchestrator as orchestrator
import opensearch_orchestrator.planning_session as planning_session
from opensearch_orchestrator.scripts.shared import Phase


class _SequencePlanner:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def __call__(self, _prompt: str) -> str:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def test_engine_workflow_it_handles_noisy_planner_outputs(monkeypatch) -> None:
    """Workflow-level engine test with stubbed planner/worker; no real external integration."""
    monkeypatch.setattr(
        planning_session,
        "preview_cap_driven_verification",
        lambda **kwargs: {
            "capabilities": ["exact", "semantic", "structured", "combined"],
            "applicable_capabilities": ["exact", "semantic", "structured", "combined"],
            "skipped_capabilities": [],
            "suggestion_meta": [],
            "selected_doc_count": 10,
            "notes": [],
        },
    )

    engine = orchestrator.create_transport_agnostic_engine(orchestrator.SessionState())
    load_result = engine.load_sample("builtin_imdb")

    assert "error" not in load_result
    assert engine.phase == Phase.GATHER_INFO
    assert load_result["text_search_required"] is True
    assert "primaryTitle" in load_result["sample_doc"]
    assert "imdb.title.basics.tsv" in load_result["status"]

    preferences = engine.set_preferences(
        budget="flexible",
        performance="balanced",
        query_pattern="balanced",
        deployment_preference="sagemaker-endpoint",
    )
    assert preferences["hybrid_weight_profile"] == "balanced"
    assert preferences["deployment_preference"] == "sagemaker-endpoint"

    planner = _SequencePlanner(
        [
            # Early, noisy finalization should be rejected internally (no user confirmation yet).
            (
                "I can finalize now.\n"
                "<planning_complete>"
                "<solution>- Retrieval Method: lexical BM25</solution>"
                "<search_capabilities>- Exact: keyword title match</search_capabilities>"
                "<keynote>- premature finalization</keynote>"
                "</planning_complete>"
            ),
            "Draft proposal ready. Ask questions or confirm to proceed.",
            (
                "Perfect, finalizing with your approved workflow.\n"
                "<planning_complete>\n"
                "<solution>\n"
                "- Retrieval Method: Hybrid Search (BM25 + Dense Vector)\n"
                "- Hybrid Weight Profile: balanced\n"
                "- Algorithm: HNSW\n"
                "- Model Deployment: OpenSearch Node (CPU)\n"
                "</solution>\n"
                "<search_capabilities>\n"
                "- Exact: precise title lookup\n"
                "- Semantic: natural-language concept retrieval\n"
                "- Structured: range and filter queries\n"
                "- Combined: hybrid query with filters\n"
                "</search_capabilities>\n"
                "<keynote>\n"
                "User approved balanced hybrid plan for IMDb-scale workflow.\n"
                "</keynote>\n"
                "</planning_complete>\n"
                "Extra assistant chatter after final block."
            ),
        ]
    )

    planning_start = asyncio.run(engine.start_planning(planning_agent=planner))
    assert planning_start["is_complete"] is False
    assert planner.calls == 2

    planning_done = asyncio.run(engine.refine_plan("yes"))
    assert planning_done["is_complete"] is True
    assert planner.calls == 3
    assert planning_done["result"] is not None
    assert "Hybrid Search" in planning_done["result"]["solution"]
    assert "- Exact:" in planning_done["result"]["search_capabilities"]
    assert "- Semantic:" in planning_done["result"]["search_capabilities"]

    captured: dict[str, str] = {}

    def _fake_worker(state: orchestrator.SessionState, context: str) -> str:
        captured["context"] = context
        return "<execution_report>{\"status\":\"success\"}</execution_report>"

    execution = asyncio.run(engine.execute_plan(worker_executor=_fake_worker))
    assert "execution_report" in execution
    assert engine.phase == Phase.DONE

    worker_context = captured["context"]
    assert "Solution:\n" in worker_context
    assert "Search Capabilities:\n" in worker_context
    assert "Keynote:\n" in worker_context
    assert "Hybrid Search" in worker_context


def test_engine_accepts_client_authored_plan_for_execution() -> None:
    """Workflow-level engine test for manual plan execution path with a fake worker."""
    engine = orchestrator.create_transport_agnostic_engine(orchestrator.SessionState())
    load_result = engine.load_sample("builtin_imdb")
    assert "error" not in load_result

    stored = engine.set_plan(
        solution="- Retrieval Method: Hybrid Search (BM25 + Dense Vector)",
        search_capabilities="- Exact: title match\n- Semantic: concept retrieval",
        keynote="Client-authored plan",
    )
    assert stored["status"] == "Plan stored."

    captured: dict[str, str] = {}

    def _fake_worker(state: orchestrator.SessionState, context: str) -> str:
        _ = state
        captured["context"] = context
        return "<execution_report>{\"status\":\"success\"}</execution_report>"

    execution = asyncio.run(engine.execute_plan(worker_executor=_fake_worker))
    assert "execution_report" in execution
    assert "Client-authored plan" in captured["context"]
