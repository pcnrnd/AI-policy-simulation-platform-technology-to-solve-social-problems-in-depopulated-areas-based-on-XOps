import { useState } from "react";
import Card from "../components/Card.jsx";
import { useAppState } from "../context/AppStateContext.jsx";

const READY_RESPONSE = `{ "status": "Ready", "waiting_for_call": true }`;

export default function DataOpsPage() {
  const { appData, addConsoleLog } = useAppState();
  const schemas = appData.metadata_schemas;

  const [activeSilo, setActiveSilo] = useState(schemas[0].silo);
  const [endpoint, setEndpoint] = useState(schemas[0].silo);
  const [responseText, setResponseText] = useState(READY_RESPONSE);

  const activeSchema = schemas.find((s) => s.silo === activeSilo) ?? schemas[0];

  const handleRunApi = () => {
    const target = schemas.find((s) => s.silo === endpoint) ?? schemas[0];
    setResponseText("Sending GET request...");
    setTimeout(() => {
      const mockResponse = {
        status: 200,
        message: "DataOps Virtualized Schema successfully fetched.",
        dataops_version: "3.0.0-R3",
        silo_silo_id: endpoint,
        dataclass_description: target.description,
        silo_virtual_table: {
          name: target.silo,
          fields: target.columns.map((c) => ({
            column: c.name,
            type: c.type,
            desc: c.description
          })),
          indexing_strategy: "PostGIS B-Tree Grid-Indexing",
          access_control: "JWT / R&D-Silo-Authenticated"
        }
      };
      setResponseText(JSON.stringify(mockResponse, null, 2));
      addConsoleLog(`INFO: DataOps REST API 호출 성공 - /api/v3/dataops/${endpoint}`);
    }, 500);
  };

  return (
    <div className="grid-details-split">
      <Card title="다기종 분산 사일로(Silo) 메타데이터 카탈로그" icon="fa-database">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          사일로 격리 환경의 데이터셋 스펙을 메타데이터 기반으로 가상화하여 연동 스키마 정보를
          단일화해서 제공합니다.
        </p>

        <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
          {schemas.map((schema) => {
            const label = schema.silo.split("_").slice(0, 2).join("_");
            const isActive = schema.silo === activeSilo;
            return (
              <button
                key={schema.silo}
                className={"btn " + (isActive ? "btn-primary" : "btn-secondary")}
                onClick={() => setActiveSilo(schema.silo)}
              >
                {label}
              </button>
            );
          })}
        </div>

        <h4 style={{ marginBottom: 12, color: "var(--accent-blue)" }}>
          📊 Silo ID: {activeSchema.silo}
        </h4>
        <p
          style={{
            fontSize: 13,
            color: "var(--text-secondary)",
            marginBottom: 16,
            fontStyle: "italic"
          }}
        >
          설명: {activeSchema.description}
        </p>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>컬럼명</th>
                <th>데이터 타입</th>
                <th>설명</th>
              </tr>
            </thead>
            <tbody>
              {activeSchema.columns.map((col) => (
                <tr key={col.name}>
                  <td>
                    <code style={{ color: "var(--accent-purple)", fontWeight: 600 }}>
                      {col.name}
                    </code>
                  </td>
                  <td>
                    <span
                      className="system-status"
                      style={{
                        padding: "1px 6px",
                        fontSize: 10,
                        backgroundColor: "rgba(59, 130, 246, 0.08)",
                        color: "var(--accent-blue)"
                      }}
                    >
                      {col.type}
                    </span>
                  </td>
                  <td>{col.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="R&D 메타데이터 데이터 조회 API 테스터" icon="fa-code">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          통합 Data API Builder 규격으로 빌드된 분산 사일로 데이터 메타데이터를 REST API 형태로 즉시
          호출할 수 있습니다.
        </p>

        <div className="slider-container">
          <label htmlFor="api-endpoint">API 엔드포인트 선택</label>
          <select
            id="api-endpoint"
            className="select-control"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
          >
            {schemas.map((s) => (
              <option key={s.silo} value={s.silo}>
                GET /api/v3/dataops/{s.silo}
              </option>
            ))}
          </select>
        </div>

        <button
          className="btn btn-primary"
          style={{ width: "100%", marginBottom: 16 }}
          onClick={handleRunApi}
        >
          <i className="fa-solid fa-paper-plane"></i> API GET Request 전송
        </button>

        <label>REST API JSON 응답 결과</label>
        <pre
          className="console-log"
          style={{
            height: 220,
            fontFamily: "monospace",
            color: "#6ee7b7",
            backgroundColor: "#040810",
            padding: 12,
            overflowY: "auto",
            whiteSpace: "pre-wrap"
          }}
        >
          {responseText}
        </pre>
      </Card>
    </div>
  );
}
