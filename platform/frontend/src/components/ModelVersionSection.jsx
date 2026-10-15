import { useState } from "react";
import Card from "./Card.jsx";
import { MODEL_TYPES, TARGET_LABELS, versionStatus } from "../lib/modelops.js";

const ACTION_NAMES = { dataset_registered: "학습데이터 등록", instance_registered: "모델 등록", pipeline_registered: "파이프라인 등록", training_saved: "학습 완료", training_failed: "학습 실패", version_activated: "버전 활성화", version_rolled_back: "이전 버전 복원", pipeline_deleted: "파이프라인 삭제" };
const metric = (value) => Number.isFinite(value) ? value.toLocaleString("ko-KR", { maximumFractionDigits: 3 }) : "—";

/**
 * 오케스트레이터 맨 아래 — 모델 종류별 등록 모델과 버전 활성화·이전 버전 복원.
 * 카탈로그·버전(상태 파생 완료)과 활성화 동작은 페이지가 소유한다(실데이터 학습 카드와 같은 상태를 쓴다).
 */
export default function ModelVersionSection({ catalog, versions, regionName, busy, selectedType, onSelectType, instanceId, onSelectInstance, onActivate }) {
  const modelInstances = catalog.instances.filter((row) => row.model_type === selectedType);
  const selectedInstance = modelInstances.find((row) => row.id === instanceId) ?? modelInstances[0];
  const selectedId = selectedInstance?.id ?? "";
  const rows = versions.filter((row) => row.instance_id === selectedId);
  // 활성화·복원 결과를 버튼이 있는 이 카드에도 표시한다(페이지 맨 위 안내와 같은 결과).
  const [result, setResult] = useState(null);
  const select = (type, id) => { setResult(null); if (type) onSelectType(type); onSelectInstance(id); };
  const decide = async (row, rollback) => setResult(await onActivate(selectedId, row, rollback));
  return <>
    <div className="grid-cols-2" role="group" aria-label="모델 종류">
      {MODEL_TYPES.map((type) => <Card key={type.id} title={type.name} icon={type.id === "factor_analysis" ? "fa-magnifying-glass-chart" : "fa-chart-line"} dataSource="api">
        <p>등록 모델 {catalog.instances.filter((row) => row.model_type === type.id).length}개</p>
        <button className="btn btn-secondary" aria-pressed={selectedType === type.id} onClick={() => select(type.id, "")}>{type.name} 버전 보기</button>
      </Card>)}
    </div>
    <Card title="모델 버전 관리" icon="fa-code-branch" dataSource="api">
      <label className="realdata-field"><span>등록 모델 선택</span><select value={selectedId} onChange={(event) => select(null, event.target.value)}>
        {!modelInstances.length && <option value="">등록된 모델 없음</option>}
        {modelInstances.map((row) => <option key={row.id} value={row.id}>{regionName(row.region_id)} · {TARGET_LABELS[row.target] ?? row.target}</option>)}
      </select></label>
      {!selectedId && <p>학습데이터를 등록하고 파이프라인을 연결해 시작하세요.</p>}
      {selectedInstance && <p>활성 버전: {selectedInstance.active_version ?? "—"} · 이전 버전: {selectedInstance.previous_version ?? "—"}</p>}
      {result && <p className={`async-feedback ${result.ok ? "is-success" : "is-error"}`} role={result.ok ? "status" : "alert"}>{result.message}</p>}
      <div className="table-container"><table><caption className="sr-only">선택한 모델의 학습 버전과 상태</caption>
        <thead><tr><th>버전</th><th>학습데이터·해시</th><th>지역·대상·기간</th><th>MAE / 기준선</th><th>상태</th><th>동작</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.id}><td>{row.id}<div>학습: {row.job_id}</div></td>
          <td>{row.dataset_id}<div title={row.dataset_hash}>{row.dataset_hash.slice(0, 12)}</div></td>
          <td>{regionName(row.region_id)} · {TARGET_LABELS[row.target] ?? row.target}<div>{row.observed_from}~{row.observed_to}</div></td>
          <td>{metric(row.metrics.mae)} / {metric(row.baseline.mae)}</td><td>{versionStatus(row.status)}</td>
          <td><button className="btn btn-secondary" disabled={busy || row.status === "active"} onClick={() => decide(row, false)}>활성화</button>{" "}
            <button className="btn btn-secondary" disabled={busy || row.status !== "previous"} onClick={() => decide(row, true)}>이전 버전 복원</button></td>
        </tr>)}</tbody>
      </table></div>
    </Card>
    <Card title="최근 운영 기록" icon="fa-clock-rotate-left" dataSource="api">
      {catalog.audit.length ? <ul>{catalog.audit.slice(0, 20).map((row) => <li key={row.id}>{ACTION_NAMES[row.action] ?? "운영 기록"} · {new Date(row.created_at).toLocaleString("ko-KR")} · {row.asset_id}</li>)}</ul>
        : <p>아직 운영 기록이 없습니다.</p>}
    </Card>
  </>;
}
