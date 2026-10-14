from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.mlops.orchestration.events import RetrainEvent
from src.mlops.orchestration.orchestrator import Orchestrator


def test_run_ids_are_unique_across_restarts_and_parallel_calls():
    def create(_):
        return Orchestrator().next_run_id()
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(create, range(200)))
    assert len(set(ids)) == len(ids)
    assert all(run_id.startswith("RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d")) for run_id in ids)


def test_rejected_candidate_artifacts_do_not_overwrite_previous_attempt():
    orchestrator = Orchestrator()
    kwargs = {"current_metrics": {"f1": 1.0, "accuracy": 1.0}, "current_version": "v1.0", "candidate_version": "v1.1"}
    first = orchestrator.handle_event(RetrainEvent("test_identity", "manual"), **kwargs)
    first_path = Path(first.artifact_path)
    first_bytes = first_path.read_bytes()
    second = orchestrator.handle_event(RetrainEvent("test_identity", "manual"), **kwargs)
    assert first.run_id != second.run_id
    assert first.artifact_path != second.artifact_path
    assert Path(second.artifact_path).is_file()
    assert first_path.read_bytes() == first_bytes
