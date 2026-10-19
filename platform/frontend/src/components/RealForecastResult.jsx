import { Line } from "react-chartjs-2";
import { formatValue, formatMonth } from "../lib/realSimulation.js";

// 정책 시뮬레이터 STAGE ③ 예측 카드의 실데이터 내용 — 관측 3점 + 예측월 선 2개, 조정 예측·기준 대비 변화, 요약.
// result 는 현재 관측 입력으로 계산한 시뮬레이션 추론이다(낡은 결과를 넘기지 않는 것은 호출부 몫).
export default function RealForecastResult({ chartData, chartOpts, result, unit, label }) {
  const deltaPercent = result.baseline_prediction
    ? ((result.delta / Math.abs(result.baseline_prediction)) * 100).toFixed(1)
    : "0.0";
  const sign = result.delta > 0 ? "+" : "";
  return (
    <>
      <div style={{ position: "relative", height: 200, width: "100%" }}>
        <Line data={chartData} options={chartOpts} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-around", marginTop: 16, textAlign: "center" }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>조정 예측 ({formatMonth(result.provenance.prediction_period)})</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-blue)" }}>
            {formatValue(result.prediction)}{unit}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>기준 대비 변화</div>
          <div className={result.delta >= 0 ? "trend-up" : "trend-down"} style={{ fontSize: 18, fontWeight: 700 }}>
            {sign}{formatValue(result.delta)}{unit} ({sign}{deltaPercent}%)
          </div>
        </div>
      </div>
      <p className="chart-summary">
        기준 예측 {formatValue(result.baseline_prediction)}{unit}, 조정 예측 {formatValue(result.prediction)}{unit}로
        모델 예측상 {formatValue(result.delta)}{unit} 변화입니다. 검증 평균 절대 오차는 {formatValue(result.provenance.metrics?.mae)}{unit}입니다.
      </p>
      {result.outside_observed_range.length > 0 && (
        <p className="async-feedback is-error" role="status">
          관측 범위 밖 입력: {result.outside_observed_range.map(label).join(", ")}. 해당 구간의 예측 정확도는 확인되지 않았습니다.
        </p>
      )}
    </>
  );
}
