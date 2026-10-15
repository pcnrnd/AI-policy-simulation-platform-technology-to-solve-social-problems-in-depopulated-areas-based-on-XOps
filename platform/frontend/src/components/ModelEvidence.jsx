import { REAL_TARGETS, observedFeatureLabel, targetUnit, formatValue, formatMonth } from "../lib/realSimulation.js";

// 계산 근거 접이식 — 두 모드에 같은 자리로 둔다. 데모는 예시 계산이라는 사실만, 실데이터는 모델·학습데이터·버전 근거를 보인다.
export default function ModelEvidence({ summary = "계산 근거", demo = false, result, regions = [], datasetName = (id) => id, slot }) {
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
        <p>활성 모델 없음 — 활성 요인분석·시뮬레이션 버전이 준비되면 모델·학습데이터와 계산 근거가 표시됩니다.</p>
      </details>
    );
  }
  const p = result.provenance;
  const unit = targetUnit(p.target);
  const evaluation = p.evaluation_period;
  return (
    <details className="real-flow-evidence" data-slot={slot}>
      <summary>{summary}</summary>
      <dl>
        <dt>지역 / 학습 대상</dt><dd>{regions.find((r) => r.id === p.region_id)?.name ?? p.region_id} / {REAL_TARGETS[p.target] ?? p.target}</dd>
        <dt>모델 종류 / 활성 버전</dt><dd>{p.model_type === "factor_analysis" ? "요인분석" : "시뮬레이션"} / {p.version_id}</dd>
        <dt>등록 모델 / 학습 작업</dt><dd>{p.instance_id} / {p.job_id}</dd>
        <dt>등록 데이터셋 / 원본</dt><dd>{datasetName(p.dataset_id)} ({p.dataset_id}) / {p.source_dataset_id}</dd>
        <dt>데이터 내용 해시</dt><dd>{p.dataset_hash}</dd>
        <dt>모델 파일 해시</dt><dd>{p.artifact_hash}</dd>
        <dt>관측 기간 / 예측월</dt><dd>{formatMonth(p.observed_from)}~{formatMonth(p.observed_to)} / {formatMonth(p.prediction_period)}</dd>
        <dt>검증 기간 / 평균 절대 오차</dt><dd>{evaluation ? `${formatMonth(evaluation.from)}~${formatMonth(evaluation.to)}` : "—"} / {formatValue(p.metrics?.mae)} {unit} (기준선 {formatValue(p.baseline?.mae)} {unit})</dd>
        <dt>입력 항목</dt><dd>{(p.features ?? []).map((feature) => observedFeatureLabel(feature, p.target, result.dong_options)).join(", ")}</dd>
        {result.inference_id && <><dt>추론 / 예측 출력</dt><dd>{result.inference_id} / {formatValue(result.prediction)} {unit}</dd></>}
      </dl>
    </details>
  );
}
