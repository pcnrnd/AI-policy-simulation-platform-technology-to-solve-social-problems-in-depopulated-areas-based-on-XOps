from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from src.api.v3 import realdata_runs
from src.mlops.monitoring import realdata_bridge
from src.realdata.binding import dataset_matches_model
from src.schemas.realdata import TrainingRunRequest

VISITOR = "namwon-nonlocal-visitors-next-month"
SALES = "namwon-observed-sales-next-month"
VISITOR_DS = {"dataset_id": "ds-00ff3407ad9f", "spec": {"model_id": VISITOR, "target": "nonlocal_visitors"}}
SALES_DS = {"dataset_id": "ds-78d6619ab243", "spec": {"model_id": SALES, "target": "observed_sales_krw"}}


def test_dataset_target_and_model_must_both_match():
    assert dataset_matches_model(VISITOR_DS, VISITOR)
    assert not dataset_matches_model(SALES_DS, VISITOR)
    assert not dataset_matches_model({"spec": {"model_id": VISITOR, "target": "observed_sales_krw"}}, VISITOR)
    assert not dataset_matches_model({}, VISITOR)


def test_catalog_ignores_mismatched_latest_job(monkeypatch):
    monkeypatch.setattr(realdata_bridge.snapshot_mod, "list_datasets", lambda: [SALES_DS, VISITOR_DS])
    monkeypatch.setattr(realdata_bridge.jobs_mod, "list_jobs", lambda _: [{"dataset_id": SALES_DS["dataset_id"], "requested_at": "2026-10-14"}])
    monkeypatch.setattr(realdata_bridge, "resolve_version", lambda _: None)
    monkeypatch.setattr(realdata_bridge.candidates_mod, "get_active", lambda _: None)
    rows = realdata_bridge.pipelines()
    assert next(row for row in rows if row["model_id"] == VISITOR)["dataset_id"] == VISITOR_DS["dataset_id"]
    monkeypatch.setattr(realdata_bridge.snapshot_mod, "list_datasets", lambda: [SALES_DS])
    assert next(row for row in realdata_bridge.pipelines() if row["model_id"] == VISITOR)["dataset_id"] is None


@pytest.mark.parametrize("datasets, status", [([SALES_DS], 400), ([], 404)])
def test_training_rejects_invalid_input_before_job_creation(monkeypatch, datasets, status):
    monkeypatch.setattr(realdata_runs.snapshot, "list_datasets", lambda: datasets)
    start = Mock()
    monkeypatch.setattr(realdata_runs.jobs, "start_training", start)
    with pytest.raises(HTTPException) as exc:
        realdata_runs.create_training_run(TrainingRunRequest(model_id=VISITOR, dataset_id=SALES_DS["dataset_id"]), {})
    assert exc.value.status_code == status
    start.assert_not_called()


def test_training_accepts_matching_input(monkeypatch):
    monkeypatch.setattr(realdata_runs.snapshot, "list_datasets", lambda: [VISITOR_DS])
    start = Mock(return_value="job-test")
    monkeypatch.setattr(realdata_runs.jobs, "start_training", start)
    monkeypatch.setattr(realdata_runs.jobs, "get_job", lambda _: {"job_id": "job-test"})
    result = realdata_runs.create_training_run(TrainingRunRequest(model_id=VISITOR, dataset_id=VISITOR_DS["dataset_id"]), {})
    assert result["status"] == "ok"
    start.assert_called_once_with(VISITOR, VISITOR_DS["dataset_id"])
