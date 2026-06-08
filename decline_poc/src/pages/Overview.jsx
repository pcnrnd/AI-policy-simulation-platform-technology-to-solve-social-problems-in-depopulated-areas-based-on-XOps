import { useAppState } from "../context/AppStateContext.jsx";
import StatCard from "../components/StatCard.jsx";
import Card from "../components/Card.jsx";

export default function Overview() {
  const { driftInjected, injectNormal, injectDrift, f1Override } = useAppState();

  const f1Value = f1Override !== null ? f1Override.toFixed(3) : "0.884";
  const f1Label =
    f1Override !== null ? "최적 (SOTA v3.1)" : "최적 (SOTA)";
  const f1Sub = f1Override !== null ? "연합 재학습 성공" : "Silo 통합 기준";

  return (
    <>
      <div className="grid-cols-3">
        <StatCard
          label="AI 예측 소멸위기 지역수"
          icon="fa-triangle-exclamation"
          value="89개소"
          footer={
            <>
              <span className="trend-up">
                <i className="fa-solid fa-caret-up"></i> 4개소
              </span>
              <span className="text-secondary">전분기 대비</span>
            </>
          }
        />
        <StatCard
          label="글로벌 모델 F1-score"
          icon="fa-bullseye"
          value={f1Value}
          footer={
            <>
              <span className="trend-up" style={{ color: "var(--accent-teal)" }}>
                <i className="fa-solid fa-circle-check"></i> {f1Label}
              </span>
              <span className="text-secondary">{f1Sub}</span>
            </>
          }
        />
        <StatCard
          label="연동 사일로 데이터 소스"
          icon="fa-network-wired"
          value="4개 실시간"
          footer={
            <>
              <span className="trend-up">
                <i className="fa-solid fa-arrow-right"></i> Active
              </span>
              <span className="text-secondary">주민·복지·산업·공간</span>
            </>
          }
        />
      </div>

      <div className="grid-details-split">
        <Card title="인구감소 R&D 3개년 마스터 마일스톤" icon="fa-road">
          <table style={{ marginTop: 10 }}>
            <thead>
              <tr>
                <th>R&D 연차</th>
                <th>핵심 기술 부문</th>
                <th>정량 목표 산출물</th>
                <th>상태</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1차년도 (25')</td>
                <td>멀티모달 공간 정보 데이터 구조 수집 설계</td>
                <td>데이터 가공 스키마 정의서, 특허 2건</td>
                <td>
                  <span
                    className="system-status"
                    style={{
                      padding: "2px 8px",
                      fontSize: 11,
                      backgroundColor: "rgba(16, 185, 129, 0.1)",
                      color: "var(--accent-teal)"
                    }}
                  >
                    완료
                  </span>
                </td>
              </tr>
              <tr>
                <td>2차년도 (25' 하)</td>
                <td>사일로 분할 가중치 모델 파라미터 수집 기술</td>
                <td>사일로 링크 구성 SW 등록 1건</td>
                <td>
                  <span
                    className="system-status"
                    style={{
                      padding: "2px 8px",
                      fontSize: 11,
                      backgroundColor: "rgba(16, 185, 129, 0.1)",
                      color: "var(--accent-teal)"
                    }}
                  >
                    완료
                  </span>
                </td>
              </tr>
              <tr>
                <td>
                  <strong>3차년도 (26' 현)</strong>
                </td>
                <td>
                  <strong>MLOps 모니터링 & 자동 재학습 오케스트레이션</strong>
                </td>
                <td>
                  <strong>모니터링 SW, 오케스트레이션 SW 등록 2건</strong>
                </td>
                <td>
                  <span
                    className="system-status retraining"
                    style={{ padding: "2px 8px", fontSize: 11 }}
                  >
                    진행 중
                  </span>
                </td>
              </tr>
              <tr>
                <td>4차년도 (27' 예정)</td>
                <td>지자체 특화 정책 추천 인공지능 시뮬레이션 고도화</td>
                <td>지자체 현장 의사결정 모델 공인인증</td>
                <td>
                  <span
                    className="system-status"
                    style={{
                      padding: "2px 8px",
                      fontSize: 11,
                      backgroundColor: "rgba(156, 163, 175, 0.1)",
                      color: "var(--text-secondary)"
                    }}
                  >
                    대기
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </Card>

        <Card
          title="빠른 제어 및 테스트"
          icon="fa-bolt"
          style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}
        >
          <div>
            <p
              style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                marginBottom: 20
              }}
            >
              통합 플랫폼 검증을 위해 실시간 데이터에 이상 현상 및 드리프트를 임의로 유입시키는 테스트
              이벤트 생성 컨트롤러입니다.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 20 }}>
              <button className="btn btn-secondary" style={{ width: "100%" }} onClick={injectNormal}>
                <i className="fa-solid fa-circle-check" style={{ color: "var(--accent-teal)" }}></i>{" "}
                정상 시나리오 주입
              </button>
              <button className="btn btn-danger" style={{ width: "100%" }} onClick={injectDrift}>
                <i className="fa-solid fa-triangle-exclamation"></i> 드리프트 (PSI {">"} 0.2) 주입
              </button>
            </div>
          </div>
          {driftInjected && (
            <div
              className="system-status"
              style={{ fontSize: 12, justifyContent: "center", display: "flex" }}
            >
              데이터 드리프트 감지됨! 슬랙 경고 발송 완료.
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
