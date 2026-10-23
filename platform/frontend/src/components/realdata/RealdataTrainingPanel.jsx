// 빠른 학습 — 재학습 파이프라인 카드 안의 단축 경로(스냅샷 생성 → 파이프라인 자동 연결 → 실행).
// 2026-10-22 별도 "데이터 학습" 카드를 합쳤다: [학습 실행]은 파이프라인 [실행]과 같은 runPipeline 이고,
// 후보 반영·복원은 ⑥ 모델 버전 관리의 활성화·이전 버전 복원과 같은 API라 표를 두지 않는다.
// 뒷단은 /api/v3/modelops/*이고, 카탈로그와 작업 폴링은 오케스트레이터 페이지가 소유한다.
// [스냅샷 생성]은 현재 모드의 원본을 동결한 등록 학습데이터를 만든다. 실데이터 모드에서는 먼저
// 실데이터 스냅샷(POST /realdata/datasets)을 만들어 그 최신 관측을 원본으로 쓴다.
import { useState } from "react";
import { apiGet, apiSend } from "../../lib/api.js";
import { alertText } from "../../lib/alertText.js";
import { JOB_STATE_NAMES, cardPipelineId, datasetMatchesInstance, displayId, instanceLabel, typeName } from "../../lib/modelops.js";
import { getRealdataToken } from "./realdataClient.jsx";

const orDash = (v) => (v === null || v === undefined || v === "" ? "–" : v);

function snapshotName(source) {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${source.name} 스냅샷 ${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`;
}

export default function RealdataTrainingPanel({ catalog, demoVisible, regionName, busy, job, onReload, onRunPipeline, onLog, onClearResult }) {
  const [selectedInstanceId, setSelectedInstanceId] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [working, setWorking] = useState(null); // "snapshot" | "train"
  const [error, setError] = useState(null);

  const instance = catalog.instances.find((row) => row.id === selectedInstanceId) ?? catalog.instances[0];
  const instanceId = instance?.id ?? "";
  const matchingDatasets = catalog.datasets.filter((row) => datasetMatchesInstance(row, instance));
  const selectedDataset = matchingDatasets.find((row) => row.id === selectedDatasetId);
  const datasetReady = Boolean(selectedDataset);
  const instanceBusy = catalog.jobs.some((row) => row.instance_id === instanceId && (row.state === "queued" || row.state === "running"));
  const locked = busy || Boolean(working);

  const handleCreateSnapshot = async () => {
    if (!instance || locked) return;
    setWorking("snapshot");
    setError(null);
    onClearResult();
    try {
      const token = await getRealdataToken();
      let sourceId = null;
      if (!demoVisible) {
        const created = await apiSend("POST", "/api/v3/realdata/datasets", { token, body: { target: instance.target } });
        if (created.status !== "ok") throw new Error(created.message ?? "실데이터 스냅샷 생성에 실패했습니다.");
        sourceId = created.data?.dataset_id ?? null;
      }
      const sources = (await apiGet("/api/v3/modelops/available-datasets", { token })).data ?? [];
      const source = sourceId
        ? sources.find((row) => row.id === sourceId)
        : sources.filter((row) => datasetMatchesInstance(row, instance)).sort((a, b) => String(b.observed_to).localeCompare(String(a.observed_to)))[0];
      if (!source) throw new Error("선택한 모델의 지역·학습 대상에 맞는 원본 데이터가 없습니다.");
      const registered = await apiSend("POST", "/api/v3/modelops/datasets", { token, body: { name: snapshotName(source), source_dataset_id: source.id, source_hash: source.content_hash } });
      // 같은 원천·내용이면 서버가 기존 등록본을 돌려준다 — 호출 전 목록(이 렌더의 catalog)에 같은 ID가 있었는지로 가른다.
      const reused = catalog.datasets.some((row) => row.id === registered.id);
      await onReload();
      setSelectedDatasetId(registered.id);
      onLog(`INFO: ${reused ? "기존 등록본 사용" : "학습데이터 스냅샷 등록"} — ${displayId(registered.id)} (${registered.observed_from}~${registered.observed_to})`);
    } catch (err) {
      setError(alertText(err, "스냅샷을 만들지 못했습니다. 잠시 뒤 다시 시도하세요."));
      onLog(`WARN: 학습데이터 스냅샷 생성 실패 — ${err?.message ?? err}`);
    }
    finally { setWorking(null); }
  };

  const handleStartTraining = async () => {
    if (!instance || !datasetReady || locked || instanceBusy) return;
    const dataset = matchingDatasets.find((row) => row.id === selectedDatasetId);
    setWorking("train");
    setError(null);
    onClearResult();
    try {
      let pipeline = catalog.pipelines.find((row) => row.instance_id === instance.id && row.dataset_id === dataset.id);
      if (!pipeline) {
        // 연결 파이프라인이 없으면 결정적 ID로 등록한다 — ① 목록에 보이고, 같은 조합이면 다시 쓴다.
        const token = await getRealdataToken();
        pipeline = await apiSend("POST", "/api/v3/modelops/pipelines", { token, body: {
          id: cardPipelineId(instance.model_type, dataset.id), name: `${typeName(instance.model_type)} 학습 (실데이터 학습 카드)`,
          model_type: instance.model_type, dataset_id: dataset.id, dataset_hash: dataset.content_hash, trigger_policy: "수동(학습 카드)"
        } });
        onLog(`INFO: 학습 카드 파이프라인 등록 — ${displayId(pipeline.id)}`);
        await onReload();
      }
      await onRunPipeline(pipeline);
    } catch (err) {
      setError(err.status === 409 ? "이미 활성 학습 작업이 있습니다." : alertText(err, "학습을 실행하지 못했습니다. 잠시 뒤 다시 시도하세요."));
      onLog(`WARN: 학습 카드 실행 실패 — ${err?.message ?? err}`);
    }
    finally { setWorking(null); }
  };

  const cardJob = job && job.instance_id === instanceId ? job : null;
  // 머리글 지역명은 선택한 등록 데이터(없으면 대상 모델)의 지역에서 읽는다. 이름을 못 찾으면 지역 코드 대신 생략.
  const regionId = selectedDataset?.region_id ?? instance?.region_id;
  const region = regionId ? regionName(regionId) : null;
  const realScope = `${region && region !== regionId ? `${region} ` : ""}행정동 실데이터`;
  return (
    <section className="realdata-panel quick-train" aria-labelledby="quick-train-title" data-values-source="api">
      <div className="quick-train-head">
        <strong id="quick-train-title"><i className="fa-solid fa-flask-vial" aria-hidden="true"></i> 빠른 학습</strong>
        <span className="quick-train-scope">
          {demoVisible ? "데모 지역 시계열 · 선형 회귀 · 후보 승인 후 반영" : `${realScope} · 선형 회귀 공동 학습 · 후보 승인 후 반영`}
        </span>
      </div>
      <div className="realdata-controls">
        <label className="realdata-field">
          <span>대상 모델</span>
          <select value={instanceId} onChange={(e) => { setSelectedInstanceId(e.target.value); setSelectedDatasetId(""); }} disabled={!catalog.instances.length}>
            {!catalog.instances.length && <option value="">등록된 모델 없음</option>}
            {catalog.instances.map((row) => <option key={row.id} value={row.id}>{instanceLabel(row, regionName)}</option>)}
          </select>
        </label>

        <label className="realdata-field">
          <span>데이터셋</span>
          <select value={selectedDatasetId} onChange={(e) => setSelectedDatasetId(e.target.value)} disabled={matchingDatasets.length === 0}>
            <option value="">선택하세요</option>
            {matchingDatasets.map((d) => (
              <option key={d.id} value={d.id}>
                {displayId(d.id)} ({orDash(d.observed_from)}~{orDash(d.observed_to)}) · {d.name}
              </option>
            ))}
          </select>
        </label>

        <button className="btn btn-secondary" onClick={handleCreateSnapshot} aria-disabled={!instance || locked}
          title={!instance ? "위 목록에서 파이프라인을 먼저 등록하세요." : "현재 원본을 동결한 학습데이터를 등록합니다."}>
          <i className={`fa-solid ${working === "snapshot" ? "fa-spinner fa-spin" : "fa-camera"}`}></i> 스냅샷 생성
        </button>
        <button
          className="btn btn-primary"
          onClick={handleStartTraining}
          aria-disabled={!instance || !datasetReady || locked || instanceBusy}
          title={!datasetReady ? "대상 모델에 맞는 스냅샷을 생성하거나 데이터셋을 선택하세요." : instanceBusy ? "현재 실행 중인 학습이 끝난 뒤 실행할 수 있습니다" : "선택한 데이터셋으로 학습을 실행합니다"}
        >
          <i className="fa-solid fa-play"></i> 학습 실행
        </button>
      </div>

      {!instance && <p className="realdata-hint">위 [파이프라인 등록]으로 대상 모델을 먼저 등록하세요.</p>}
      {instance && (
        <p className="realdata-hint">
          활성 버전: <code title={instance.active_version ?? undefined}>{orDash(displayId(instance.active_version))}</code>
          {" "}· 학습이 끝난 후보는 아래 [모델 버전 관리]에서 활성화합니다.
        </p>
      )}
      {cardJob && (
        <p className="realdata-hint" role="status">
          작업 <code title={cardJob.id}>{displayId(cardJob.id)}</code> — 상태 <strong>{JOB_STATE_NAMES[cardJob.state] ?? cardJob.state}</strong>
          {cardJob.error && ` (${cardJob.error})`}
        </p>
      )}
      {error && <p className="realdata-error" role="alert">{error}</p>}
    </section>
  );
}
