import Card from "./Card.jsx";
import ConsoleLog from "./ConsoleLog.jsx";
import { TARGET_LABELS, auditLogLines, displayId, instanceLabel, typeName, versionStatus } from "../lib/modelops.js";
import { formatUnitValue, targetUnit } from "../lib/realSimulation.js";

// MAE·기준선은 학습 대상 단위 규칙(시뮬레이터 ①과 같은 formatUnitValue)으로 보인다.
const metric = (value, target) => formatUnitValue(value, targetUnit(target));
const head = (text, length) => <code title={text}>{String(text ?? "").slice(0, length)}</code>;
const shortCode = (id) => <code title={id}>{displayId(id)}</code>;

/**
 * 오케스트레이터 맨 아래 — 등록 모델(종류·지역·학습 대상)별 버전 활성화·이전 버전 복원과 최근 운영 기록.
 * 카탈로그·버전(상태 파생 완료)과 활성화 동작은 페이지가 소유한다(실데이터 학습 카드와 같은 상태를 쓴다).
 */
export default function ModelVersionSection({ catalog, versions, regionName, busy, instanceId, onSelectInstance, onActivate, versionResult, onClearResult }) {
  const selectedInstance = catalog.instances.find((row) => row.id === instanceId) ?? catalog.instances[0];
  const selectedId = selectedInstance?.id ?? "";
  const rows = versions.filter((row) => row.instance_id === selectedId);
  // 활성화·복원 결과는 페이지 공유 상태 — 이 카드에서 시작한 동작(origin "versions")일 때만 여기 표시한다.
  const result = versionResult?.origin === "versions" ? versionResult : null;
  const select = (id) => { onClearResult(); onSelectInstance(id); };
  const decide = (row, rollback) => onActivate(selectedId, row, rollback);
  const auditLines = auditLogLines(catalog.audit);
  return <>
    <Card title="모델 버전 관리" icon="fa-code-branch" className="page-section realdata-panel" dataSource="api">
      <div className="realdata-controls">
        <label className="realdata-field"><span>등록 모델 선택</span><select value={selectedId} onChange={(event) => select(event.target.value)}>
          {!catalog.instances.length && <option value="">등록된 모델 없음</option>}
          {catalog.instances.map((row) => <option key={row.id} value={row.id}>{instanceLabel(row, regionName)}</option>)}
        </select></label>
      </div>
      {!selectedId && <p className="realdata-hint">학습데이터를 등록하고 파이프라인을 연결해 시작하세요.</p>}
      {selectedInstance && <p className="realdata-hint">종류: {typeName(selectedInstance.model_type)} · 활성 버전: {selectedInstance.active_version ? shortCode(selectedInstance.active_version) : "—"} · 이전 버전: {selectedInstance.previous_version ? shortCode(selectedInstance.previous_version) : "—"}</p>}
      {result && <p className={`async-feedback ${result.ok ? "is-success" : "is-error"}`} role={result.ok ? "status" : "alert"}>{result.message}</p>}
      <div className="table-container"><table><caption className="sr-only">선택한 모델의 학습 버전과 상태</caption>
        <thead><tr><th scope="col">버전</th><th scope="col">학습데이터·해시</th><th scope="col">지역·대상·기간</th><th scope="col" className="cell-num">MAE / 기준선</th><th scope="col">상태</th><th scope="col" className="cell-actions">동작</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.id}><td>{shortCode(row.id)}<div>학습: {shortCode(row.job_id)}</div></td>
          <td>{shortCode(row.dataset_id)}<div>{head(row.dataset_hash, 12)}</div></td>
          <td>{regionName(row.region_id)} · {TARGET_LABELS[row.target] ?? row.target}<div>{row.observed_from}~{row.observed_to}</div></td>
          <td className="cell-num">{metric(row.metrics.mae, row.target)} / {metric(row.baseline.mae, row.target)}</td>
          <td><span className="system-status" style={{ padding: "1px 8px", fontSize: 10, whiteSpace: "nowrap" }}>{versionStatus(row.status)}</span></td>
          <td className="cell-actions" style={{ whiteSpace: "nowrap" }}><button className="btn btn-secondary" disabled={busy || row.status === "active"} onClick={() => decide(row, false)}>활성화</button>{" "}
            <button className="btn btn-secondary" disabled={busy || row.status !== "previous"} onClick={() => decide(row, true)}>이전 버전 복원</button></td>
        </tr>)}</tbody>
      </table></div>
    </Card>
    <Card title="최근 운영 기록" icon="fa-clock-rotate-left" className="page-section" dataSource="api">
      {auditLines.length ? <ConsoleLog logs={auditLines} height={220} /> : (
        <p className="pipeline-empty">
          <i className="fa-solid fa-circle-info" aria-hidden="true"></i>{" "}아직 운영 기록이 없습니다.
        </p>
      )}
    </Card>
  </>;
}
