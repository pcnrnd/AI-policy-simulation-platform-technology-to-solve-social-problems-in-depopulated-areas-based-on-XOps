import { REAL_TARGETS, observedFeatureLabel, targetUnit, formatValue, formatMonth, contributionSplit } from "../lib/realSimulation.js";

// 학습 시 비교한 기준선 방법 — 알려진 것만 한글로 보인다(모르는 이름은 표시하지 않는다).
const BASELINE_NAMES = { yoy_or_prev_month: "전년 같은 달·전월 관측값" };

// 계산 근거 접이식 — 두 모드에 같은 자리로 둔다. 데모는 예시 계산이라는 사실만, 실데이터는 모델·학습데이터·버전 근거를 보인다.
// 추론 결과(result.contributions)가 있으면 예측 입력 기여도(예측을 높인/낮춘 입력)도 함께 보인다.
// state: 근거가 없을 때 이유 — "loading"(조회 중) | "error"(조회 실패) | 없음(활성 모델 없음 또는 emptyText).
export default function ModelEvidence({ summary = "계산 근거", demo = false, result, regions = [], datasetName = (id) => id, slot, state, emptyText, label: featureLabel }) {
  if (demo) {
    return (
      <details className="real-flow-evidence" data-slot={slot}>
        <summary>{summary}</summary>
        <dl>
          <dt>데이터</dt><dd>예시 사례 데이터</dd>
          <dt>계산</dt><dd>화면 예시 계산</dd>
          <dt>모델 버전</dt><dd>없음(예시)</dd>
        </dl>
      </details>
    );
  }
  if (!result) {
    return (
      <details className="real-flow-evidence" data-slot={slot}>
        <summary>{summary}</summary>
        {emptyText ? (
          <p>{emptyText}</p>
        ) : state === "loading" ? (
          <p role="status"><i className="fa-solid fa-spinner fa-spin" aria-hidden="true"></i> 모델과 관측 입력을 확인하는 중입니다…</p>
        ) : state === "error" ? (
          <p>모델 정보를 불러오지 못했습니다. [정책 추천 도출]을 눌러 다시 확인하거나 잠시 뒤 다시 시도하세요.</p>
        ) : (
          <p>활성 모델 없음 — 시뮬레이션 활성 버전이 준비되면 모델·학습데이터와 계산 근거가 표시됩니다.</p>
        )}
      </details>
    );
  }
  const p = result.provenance;
  const unit = targetUnit(p.target);
  const evaluation = p.evaluation_period;
  const label = featureLabel ?? ((feature) => observedFeatureLabel(feature, p.target, result.dong_options));
  const split = result.contributions ? contributionSplit(result, label) : null;
  const rowsText = (rows) => rows.length ? rows.map((row) => `${row.label} ${row.value > 0 ? "+" : ""}${formatValue(row.value)}${unit}`).join(", ") : "없음";
  const baselineName = BASELINE_NAMES[p.baseline?.name];
  return (
    <details className="real-flow-evidence" data-slot={slot}>
      <summary>{summary}</summary>
      <dl>
        <dt>지역 / 학습 대상</dt><dd>{regions.find((r) => r.id === p.region_id)?.name ?? p.region_id} / {REAL_TARGETS[p.target] ?? p.target}</dd>
        <dt>모델 종류 / 활성 버전</dt><dd>시뮬레이션 / {p.version_id}</dd>
        <dt>등록 모델 / 학습 작업</dt><dd>{p.instance_id} / {p.job_id}</dd>
        <dt>등록 데이터셋 / 원본</dt><dd>{datasetName(p.dataset_id)} ({p.dataset_id}) / {p.source_dataset_id}</dd>
        <dt>데이터 내용 해시</dt><dd>{p.dataset_hash}</dd>
        <dt>모델 파일 해시</dt><dd>{p.artifact_hash}</dd>
        <dt>관측 기간 / 예측월</dt><dd>{formatMonth(p.observed_from)}~{formatMonth(p.observed_to)} / {formatMonth(p.prediction_period)}</dd>
        <dt>검증 기간 / 평균 절대 오차</dt><dd>{evaluation ? `${formatMonth(evaluation.from)}~${formatMonth(evaluation.to)}` : "—"} / {formatValue(p.metrics?.mae)} {unit} (기준선 {formatValue(p.baseline?.mae)} {unit}{baselineName ? ` · ${baselineName}` : ""})</dd>
        <dt>입력 항목</dt><dd>{(p.features ?? []).map(label).join(", ")}</dd>
        {result.inference_id && <><dt>추론 / 예측 출력</dt><dd>{result.inference_id} / {formatValue(result.prediction)} {unit}</dd></>}
        {split && (
          <>
            <dt>예측 입력 기여도</dt><dd>기준값 {formatValue(result.intercept)} {unit} + 입력별 기여도 합 = 예측 {formatValue(result.prediction)} {unit}</dd>
            <dt>예측을 높인 입력</dt><dd>{rowsText(split.positive)}</dd>
            <dt>예측을 낮춘 입력</dt><dd>{rowsText(split.negative)}</dd>
          </>
        )}
      </dl>
    </details>
  );
}
