// 인구감소 R&D 통합 플랫폼 - Core Business Logic
document.addEventListener("DOMContentLoaded", async () => {
  
  // ==========================================
  // 1. Global State Management
  // ==========================================
  let appData = null;
  let activeTab = "tab-overview";
  
  // Simulation Variables
  let currentRegion = null;
  let welfareWeight = 50;
  let industryWeight = 50;
  let housingWeight = 50;
  
  // Chart instances for global disposal/updating
  let chartDrift = null;
  let chartMetrics = null;
  let chartShap = null;
  let chartPredict = null;
  
  // Pipeline State
  let pipelineRunning = false;
  let currentPipelineStep = 0;
  let pipelineTimer = null;
  
  // Alert Status
  let driftInjected = false;
  
  // ==========================================
  // 2. Fetch Initial Mock Data
  // ==========================================
  async function loadData() {
    try {
      const response = await fetch("assets/mock_data.json");
      appData = await response.json();
      
      // Select default region
      currentRegion = appData.regions[0];
      
      // Initialize systems
      initTabNavigation();
      initMLOpsMonitor();
      initDataOpsCatalog();
      initPolicySimulator();
      initAutoReporter();
      initInjectControls();
      
      addConsoleLog("INFO: 모든 R&D 데이터 모델 및 스키마 자산이 성공적으로 연동되었습니다.");
    } catch (error) {
      console.error("데이터 로드 실패:", error);
      addConsoleLog("ERROR: R&D 자산 mock_data.json 파싱 실패.", true);
    }
  }

  // ==========================================
  // 3. Tab Routing Logic
  // ==========================================
  function initTabNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const headerTitle = document.getElementById("header-page-title");
    
    navItems.forEach(item => {
      item.addEventListener("click", () => {
        const targetTab = item.getAttribute("data-tab");
        if (targetTab === activeTab) return;
        
        // Remove active from previous
        document.querySelector(".nav-item.active").classList.remove("active");
        document.querySelector(".page-tab.active").classList.remove("active");
        
        // Add active to current
        item.classList.add("active");
        const tabEl = document.getElementById(targetTab);
        tabEl.classList.add("active");
        
        activeTab = targetTab;
        headerTitle.textContent = item.querySelector("span").textContent;
        
        // Trigger resize to fix chart layouts when hidden tabs become visible
        setTimeout(() => {
          window.dispatchEvent(new Event('resize'));
        }, 100);
      });
    });
  }

  // ==========================================
  // 4. Console Logger Terminal Utilities
  // ==========================================
  function addConsoleLog(message, isError = false, isWarn = false) {
    const consoleEl = document.getElementById("orchestrator-console");
    if (!consoleEl) return;
    
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    
    let typeClass = "";
    if (isError) typeClass = "log-err";
    else if (isWarn) typeClass = "log-warn";
    
    const logNode = document.createElement("div");
    logNode.className = "log-entry";
    logNode.innerHTML = `<span class="log-time">[${timeStr}]</span><span class="${typeClass}">${message}</span>`;
    
    consoleEl.appendChild(logNode);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  // ==========================================
  // 5. MLOps Performance Monitor Page
  // ==========================================
  function initMLOpsMonitor() {
    if (!appData) return;
    
    // Draw Drift Chart
    const ctxDrift = document.getElementById("chart-drift-distribution").getContext("2d");
    chartDrift = new Chart(ctxDrift, {
      type: "bar",
      data: {
        labels: appData.drift_distribution.buckets,
        datasets: [
          {
            label: "참조 분포 (Reference Dataset)",
            data: appData.drift_distribution.reference,
            backgroundColor: "rgba(59, 130, 246, 0.4)",
            borderColor: "rgba(59, 130, 246, 1)",
            borderWidth: 1.5,
            borderRadius: 4
          },
          {
            label: "실시간 유입 데이터 분포",
            data: driftInjected ? appData.drift_distribution.current_drifted : appData.drift_distribution.current_normal,
            backgroundColor: driftInjected ? "rgba(239, 68, 68, 0.65)" : "rgba(16, 185, 129, 0.5)",
            borderColor: driftInjected ? "rgba(239, 68, 68, 1)" : "rgba(16, 185, 129, 1)",
            borderWidth: 1.5,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
          x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
        },
        plugins: {
          legend: { labels: { color: "#f3f4f6", font: { family: "Inter" } } }
        }
      }
    });
    
    // Draw 6 Core MLOps Metrics History Chart
    const ctxMetrics = document.getElementById("chart-core-metrics").getContext("2d");
    chartMetrics = new Chart(ctxMetrics, {
      type: "line",
      data: {
        labels: appData.metrics_history.timestamps,
        datasets: [
          {
            label: "Accuracy",
            data: appData.metrics_history.accuracy,
            borderColor: "rgba(16, 185, 129, 1)",
            backgroundColor: "rgba(16, 185, 129, 0.05)",
            fill: true,
            tension: 0.35,
            borderWidth: 2
          },
          {
            label: "F1-Score",
            data: appData.metrics_history.f1,
            borderColor: "rgba(59, 130, 246, 1)",
            borderWidth: 2,
            pointStyle: "circle",
            tension: 0.35
          },
          {
            label: "MSE",
            data: appData.metrics_history.mse,
            borderColor: "rgba(239, 68, 68, 1)",
            borderWidth: 1.5,
            borderDash: [5, 5],
            tension: 0.35
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
          x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
        },
        plugins: {
          legend: { labels: { color: "#f3f4f6" } }
        }
      }
    });
    
    // Draw SHAP Feature Importance
    const ctxShap = document.getElementById("chart-shap-importance").getContext("2d");
    chartShap = new Chart(ctxShap, {
      type: "bar",
      data: {
        labels: appData.shap_features.map(f => f.feature.split(" (")[0]),
        datasets: [{
          label: "SHAP 기여도 기여 지수 (음수일수록 유출 기여)",
          data: appData.shap_features.map(f => f.value),
          backgroundColor: appData.shap_features.map(f => f.value > 0 ? "rgba(16, 185, 129, 0.65)" : "rgba(239, 68, 68, 0.65)"),
          borderColor: appData.shap_features.map(f => f.value > 0 ? "rgba(16, 185, 129, 1)" : "rgba(239, 68, 68, 1)"),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
          y: { grid: { display: false }, ticks: { color: "#f3f4f6", font: { size: 10 } } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  function updateMLOpsMonitorData() {
    if (!appData) return;
    
    const psiVal = document.getElementById("monitor-psi-val");
    const driftStatus = document.getElementById("monitor-drift-status");
    const statusText = document.getElementById("system-status-text");
    const statusIndicator = document.getElementById("system-status-indicator");
    const alertDot = document.getElementById("system-alert-dot");
    const driftCard = document.getElementById("monitor-drift-card");
    const overviewF1Val = document.getElementById("overview-f1-val");
    const overviewF1Footer = document.getElementById("overview-f1-footer");
    
    if (driftInjected) {
      psiVal.textContent = "0.384";
      psiVal.style.color = "var(--accent-red)";
      driftStatus.textContent = "위험 (Drift)";
      driftStatus.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
      driftStatus.style.color = "var(--accent-red)";
      
      statusText.textContent = "이상 현상: 데이터 드리프트 감지 (PSI: 0.384)";
      statusIndicator.classList.add("drift-alert");
      statusIndicator.parentElement.className = "system-status drift-alert";
      
      alertDot.style.display = "block";
      driftCard.className = "card glow-red";
      
      // Update chart dataset
      chartDrift.data.datasets[1].data = appData.drift_distribution.current_drifted;
      chartDrift.data.datasets[1].backgroundColor = "rgba(239, 68, 68, 0.65)";
      chartDrift.data.datasets[1].borderColor = "rgba(239, 68, 68, 1)";
      
      // Outlier Table update
      document.getElementById("monitor-outlier-val").textContent = "3건";
      document.getElementById("monitor-outlier-val").style.color = "var(--accent-red)";
      
      const outlierTbody = document.getElementById("outlier-table-body");
      outlierTbody.innerHTML = `
        <tr style="animation: fadeIn 0.5s;">
          <td style="color: var(--accent-red);">13:02</td>
          <td>의성군 데이터 (인구비율)</td>
          <td>3.45</td>
          <td><span class="outlier-tag">Outlier</span></td>
        </tr>
        <tr style="animation: fadeIn 0.5s;">
          <td style="color: var(--accent-red);">13:01</td>
          <td>고흥군 데이터 (일자리수)</td>
          <td>2.89</td>
          <td><span class="outlier-tag">Outlier</span></td>
        </tr>
        <tr>
          <td style="color: var(--text-secondary);">12:34</td>
          <td>의성군 데이터</td>
          <td>1.24</td>
          <td><span class="system-status" style="padding: 1px 6px; font-size: 10px; background-color: rgba(16, 185, 129, 0.1); color: var(--accent-teal);">Normal</span></td>
        </tr>
      `;
    } else {
      psiVal.textContent = "0.045";
      psiVal.style.color = "var(--accent-teal)";
      driftStatus.textContent = "정상";
      driftStatus.style.backgroundColor = "rgba(16, 185, 129, 0.1)";
      driftStatus.style.color = "var(--accent-teal)";
      
      statusText.textContent = "모델 모니터링 활성 (정상)";
      statusIndicator.parentElement.className = "system-status";
      
      alertDot.style.display = "none";
      driftCard.className = "card";
      
      // Update chart dataset
      chartDrift.data.datasets[1].data = appData.drift_distribution.current_normal;
      chartDrift.data.datasets[1].backgroundColor = "rgba(16, 185, 129, 0.5)";
      chartDrift.data.datasets[1].borderColor = "rgba(16, 185, 129, 1)";
      
      // Reset metrics
      document.getElementById("monitor-outlier-val").textContent = "0건";
      document.getElementById("monitor-outlier-val").style.color = "var(--text-primary)";
      
      document.getElementById("monitor-acc-val").textContent = "0.892";
      overviewF1Val.textContent = "0.884";
      overviewF1Footer.innerHTML = `<span class="trend-up" style="color: var(--accent-teal);"><i class="fa-solid fa-circle-check"></i> 최적 (SOTA)</span><span class="text-secondary">Silo 통합 기준</span>`;
      
      const outlierTbody = document.getElementById("outlier-table-body");
      outlierTbody.innerHTML = `
        <tr>
          <td style="color: var(--text-secondary);">12:34</td>
          <td>의성군 데이터</td>
          <td>1.24</td>
          <td><span class="system-status" style="padding: 1px 6px; font-size: 10px; background-color: rgba(16, 185, 129, 0.1); color: var(--accent-teal);">Normal</span></td>
        </tr>
        <tr>
          <td style="color: var(--text-secondary);">11:20</td>
          <td>고흥군 데이터</td>
          <td>0.98</td>
          <td><span class="system-status" style="padding: 1px 6px; font-size: 10px; background-color: rgba(16, 185, 129, 0.1); color: var(--accent-teal);">Normal</span></td>
        </tr>
      `;
    }
    
    chartDrift.update();
  }

  // ==========================================
  // 6. DataOps Schema Explorer Page
  // ==========================================
  function initDataOpsCatalog() {
    if (!appData) return;
    
    const btnsContainer = document.getElementById("dataops-silo-btns");
    const apiSelect = document.getElementById("api-endpoint-select");
    
    btnsContainer.innerHTML = "";
    apiSelect.innerHTML = "";
    
    appData.metadata_schemas.forEach((schema, idx) => {
      // 1. Schema Catalog Buttons
      const btn = document.createElement("button");
      btn.className = idx === 0 ? "btn btn-primary" : "btn btn-secondary";
      btn.textContent = schema.silo.split("_").slice(0, 2).join("_");
      btn.addEventListener("click", () => {
        // Active button toggle
        btnsContainer.querySelector(".btn-primary").className = "btn btn-secondary";
        btn.className = "btn btn-primary";
        renderSiloSchema(schema);
      });
      btnsContainer.appendChild(btn);
      
      // 2. API Endpoint Selector options
      const opt = document.createElement("option");
      opt.value = schema.silo;
      opt.textContent = `GET /api/v3/dataops/${schema.silo}`;
      apiSelect.appendChild(opt);
    });
    
    // Render first silo catalog by default
    renderSiloSchema(appData.metadata_schemas[0]);
    
    // API GET Request Button binding
    document.getElementById("btn-run-api").addEventListener("click", () => {
      const selectedSilo = apiSelect.value;
      const targetSchema = appData.metadata_schemas.find(s => s.silo === selectedSilo);
      
      const responseView = document.getElementById("api-response-view");
      responseView.textContent = "Sending GET request...";
      
      setTimeout(() => {
        const mockResponse = {
          "status": 200,
          "message": "DataOps Virtualized Schema successfully fetched.",
          "dataops_version": "3.0.0-R3",
          "silo_silo_id": selectedSilo,
          "dataclass_description": targetSchema.description,
          "silo_virtual_table": {
            "name": targetSchema.silo,
            "fields": targetSchema.columns.map(c => ({
              "column": c.name,
              "type": c.type,
              "desc": c.description
            })),
            "indexing_strategy": "PostGIS B-Tree Grid-Indexing",
            "access_control": "JWT / R&D-Silo-Authenticated"
          }
        };
        responseView.textContent = JSON.stringify(mockResponse, null, 2);
        addConsoleLog(`INFO: DataOps REST API 호출 성공 - /api/v3/dataops/${selectedSilo}`);
      }, 500);
    });
  }

  function renderSiloSchema(schema) {
    document.getElementById("dataops-silo-title").textContent = `📊 Silo ID: ${schema.silo}`;
    document.getElementById("dataops-silo-desc").textContent = `설명: ${schema.description}`;
    
    const tbody = document.getElementById("dataops-schema-tbody");
    tbody.innerHTML = "";
    
    schema.columns.forEach(col => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code style="color: var(--accent-purple); font-weight:600;">${col.name}</code></td>
        <td><span class="system-status" style="padding: 1px 6px; font-size:10px; background-color:rgba(59, 130, 246, 0.08); color: var(--accent-blue);">${col.type}</span></td>
        <td>${col.description}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ==========================================
  // 7. MLOps Retraining Orchestration System
  // ==========================================
  const pipelineSteps = [
    { name: "node-event", desc: "1. Event Trigger", log: "ALERT: 데이터 드리프트 감지 이벤트 수신. (PSI: 0.384 > 임계치 0.20) 재학습 자동 스케줄 파이프라인 트리거 완료.", warn: true },
    { name: "node-prep", desc: "2. Data Prep", log: "INFO: 분산 사일로 데이터 수집 및 정제 처리 가동. 4개 사일로(주민·복지·산업·공간) 메타데이터 검증 완료. 데이터셋 로딩 중..." },
    { name: "node-train", desc: "3. Retraining", log: "INFO: Flower 연합학습 엔진 기반 분산 학습 개시. 6개 클라이언트 사일로에서 분할 가중치 병렬 트레이닝 시작 (Differential Privacy 적용)." },
    { name: "node-eval", desc: "4. Evaluation", log: "INFO: 학습 완료. 모델 메트릭 자동 비교 연산 수행. [SOTA 검증성공: Accuracy 기존 0.892 -> 신규 0.925 (+3.3% 향상, 승격 DoD 1.5% 돌파)]." },
    { name: "node-deploy", desc: "5. Canary Deploy", log: "INFO: 신규 모델 카나리 점진적 배포 단계 진입. 트래픽 10% 자동 유입 및 롤백 임계 상태 모니터링 개시. 지연속도 120ms 양호." },
    { name: "node-rollback", desc: "6. SOTA Promoted", log: "SUCCESS: 최적 성능 모델(SOTA)로 최종 승급 승격 완료! Helm Chart 및 모델 레지스트리 정보 자동 갱신 완료." }
  ];

  document.getElementById("btn-trigger-orchestrator").addEventListener("click", () => {
    if (pipelineRunning) return;
    startOrchestratorPipeline();
  });
  
  document.getElementById("btn-reset-pipeline").addEventListener("click", () => {
    resetPipelineUI();
    addConsoleLog("INFO: MLOps 재학습 파이프라인이 초기화되었습니다.");
  });

  function resetPipelineUI() {
    clearTimeout(pipelineTimer);
    pipelineRunning = false;
    currentPipelineStep = 0;
    
    // Clear styles
    document.querySelectorAll(".pipeline-node").forEach(node => {
      node.className = "pipeline-node";
    });
    document.querySelectorAll(".pipeline-connector").forEach(conn => {
      conn.className = "pipeline-connector";
    });
  }

  function startOrchestratorPipeline() {
    resetPipelineUI();
    pipelineRunning = true;
    
    const statusText = document.getElementById("system-status-text");
    const statusIndicator = document.getElementById("system-status-indicator");
    
    statusText.textContent = "자동 재학습 및 배포 파이프라인 수행 중...";
    statusIndicator.parentElement.className = "system-status retraining";
    
    addConsoleLog("INFO: 이벤트 기반 MLOps 오케스트레이션 자동 파이프라인 실행 시작.");
    executePipelineStep();
  }

  function executePipelineStep() {
    if (currentPipelineStep >= pipelineSteps.length) {
      // Completed fully!
      pipelineRunning = false;
      addConsoleLog("SUCCESS: 모든 모델 성능 모니터링 & 복원 오케스트레이션 수행이 완료되었습니다.");
      
      const statusText = document.getElementById("system-status-text");
      const statusIndicator = document.getElementById("system-status-indicator");
      
      // Update global accuracy indicators to show SOTA Succeeded
      document.getElementById("monitor-acc-val").textContent = "0.925";
      document.getElementById("monitor-f1-val").textContent = "F1: 0.916";
      document.getElementById("overview-f1-val").textContent = "0.916";
      document.getElementById("overview-f1-footer").innerHTML = `<span class="trend-up" style="color: var(--accent-teal);"><i class="fa-solid fa-circle-check"></i> 최적 (SOTA v3.1)</span><span class="text-secondary">연합 재학습 성공</span>`;
      
      // Drift alerts reset
      driftInjected = false;
      updateMLOpsMonitorData();
      
      return;
    }
    
    const step = pipelineSteps[currentPipelineStep];
    const nodeEl = document.getElementById(step.name);
    
    // Set active
    nodeEl.className = "pipeline-node active";
    addConsoleLog(step.log, step.warn || false, false);
    
    // Connect connector
    if (currentPipelineStep > 0) {
      const prevStep = pipelineSteps[currentPipelineStep - 1];
      const prevNodeEl = document.getElementById(prevStep.name);
      prevNodeEl.className = "pipeline-node success";
      
      const connId = `conn-${prevStep.name.split("-")[1]}-${step.name.split("-")[1]}`;
      const connEl = document.getElementById(connId);
      if (connEl) connEl.className = "pipeline-connector success";
    }
    
    // Set up next step
    currentPipelineStep++;
    
    // Trigger connector loader for visual flow
    if (currentPipelineStep < pipelineSteps.length) {
      const nextStep = pipelineSteps[currentPipelineStep];
      const connId = `conn-${step.name.split("-")[1]}-${nextStep.name.split("-")[1]}`;
      const connEl = document.getElementById(connId);
      if (connEl) connEl.className = "pipeline-connector active";
    }
    
    pipelineTimer = setTimeout(executePipelineStep, 2500);
  }

  // ==========================================
  // 8. Spatial Mapping & Policy Simulator Page
  // ==========================================
  function initPolicySimulator() {
    if (!appData) return;
    
    // Initialize Leaflet Map
    const map = L.map('map').setView([35.8, 128.0], 7.5);
    
    // Beautiful Custom Dark theme tiles (Futuristic look)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }).addTo(map);
    
    // Add markers for all 6 regions
    appData.regions.forEach(region => {
      // Define color based on riskIndex (lower is more dangerous)
      const color = region.riskIndex < 0.15 ? "var(--accent-red)" : "var(--accent-orange)";
      
      const marker = L.circleMarker([region.lat, region.lng], {
        radius: 10,
        fillColor: color,
        color: "#ffffff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
      }).addTo(map);
      
      // Popup binding
      marker.bindPopup(`
        <div style="font-family: 'Inter', sans-serif;">
          <h4 style="margin-bottom:6px; color: ${color};">${region.name}</h4>
          <p style="font-size:12px; margin: 2px 0;">인구수: <strong>${region.population.toLocaleString()}명</strong></p>
          <p style="font-size:12px; margin: 2px 0;">위험지수: <strong>${region.riskIndex}</strong> (소멸위기)</p>
          <p style="font-size:11px; margin-top:6px; color: var(--text-secondary);">클릭하여 정책 가이드를 설정하세요.</p>
        </div>
      `);
      
      marker.on("click", () => {
        selectRegion(region);
      });
    });
    
    // Simulator sliders
    const sliderWelfare = document.getElementById("slider-welfare");
    const sliderIndustry = document.getElementById("slider-industry");
    const sliderHousing = document.getElementById("slider-housing");
    
    const valWelfare = document.getElementById("val-slider-welfare");
    const valIndustry = document.getElementById("val-slider-industry");
    const valHousing = document.getElementById("val-slider-housing");
    
    sliderWelfare.addEventListener("input", (e) => {
      welfareWeight = parseInt(e.target.value);
      valWelfare.textContent = `${welfareWeight}%`;
      runSimulation();
    });
    
    sliderIndustry.addEventListener("input", (e) => {
      industryWeight = parseInt(e.target.value);
      valIndustry.textContent = `${industryWeight}%`;
      runSimulation();
    });
    
    sliderHousing.addEventListener("input", (e) => {
      housingWeight = parseInt(e.target.value);
      valHousing.textContent = `${housingWeight}%`;
      runSimulation();
    });
    
    // Draw Prediction Line Chart
    const ctxPredict = document.getElementById("chart-population-prediction").getContext("2d");
    chartPredict = new Chart(ctxPredict, {
      type: "line",
      data: {
        labels: ["2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035"],
        datasets: [
          {
            label: "자연 감소 예측 추이 (Base Model)",
            data: [],
            borderColor: "rgba(239, 68, 68, 0.65)",
            borderDash: [5, 5],
            fill: false,
            tension: 0.2
          },
          {
            label: "정책 시뮬레이션 적용 예측 추이",
            data: [],
            borderColor: "rgba(59, 130, 246, 1)",
            backgroundColor: "rgba(59, 130, 246, 0.05)",
            fill: true,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
          x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
        },
        plugins: {
          legend: { labels: { color: "#f3f4f6", boxWidth: 12 } }
        }
      }
    });
    
    // Select default
    selectRegion(currentRegion);
  }

  function selectRegion(region) {
    currentRegion = region;
    document.getElementById("sim-region-name").textContent = region.name;
    
    // If we're on reporting tab, sync it
    const repRegionSelect = document.getElementById("reporter-region-select");
    if (repRegionSelect) repRegionSelect.value = region.id;
    
    runSimulation();
  }

  function runSimulation() {
    if (!currentRegion) return;
    
    // Simple dynamic mathematical simulation based on regional baseline parameters
    const basePop = currentRegion.population;
    const welfareModifier = currentRegion.policyImpacts.welfare * (welfareWeight / 100);
    const industryModifier = currentRegion.policyImpacts.industry * (industryWeight / 100);
    const housingModifier = currentRegion.policyImpacts.housing * (housingWeight / 100);
    
    const combinedModifier = (welfareModifier + industryModifier + housingModifier); // e.g. up to ~0.5 Max
    
    const baseTrend = [];
    const simulatedTrend = [];
    
    let basePopTracker = basePop;
    let simPopTracker = basePop;
    
    // Base natural decline rate (losing ~2% population per year)
    const naturalDecline = 0.98;
    
    for (let i = 0; i < 10; i++) {
      basePopTracker = Math.round(basePopTracker * naturalDecline);
      baseTrend.push(basePopTracker);
      
      // Simulated decline rate (mitigated by policy modifier)
      // policy can increase retention up to positive growth rate
      const growthModifier = combinedModifier * 0.06; // up to ~3% growth
      const currentDeclineRate = naturalDecline + growthModifier;
      
      simPopTracker = Math.round(simPopTracker * currentDeclineRate);
      simulatedTrend.push(simPopTracker);
    }
    
    // Update chart
    chartPredict.data.datasets[0].data = baseTrend;
    chartPredict.data.datasets[1].data = simulatedTrend;
    chartPredict.update();
    
    // Update stats
    const finalPop = simulatedTrend[9];
    const growthPercent = (((finalPop - basePop) / basePop) * 100).toFixed(1);
    
    document.getElementById("sim-result-pop").textContent = `${finalPop.toLocaleString()}명`;
    
    const growthEl = document.getElementById("sim-result-birth");
    growthEl.textContent = `${growthPercent > 0 ? "+" : ""}${growthPercent}%`;
    growthEl.className = growthPercent >= 0 ? "trend-up" : "trend-down";
  }

  // ==========================================
  // 9. Automated Reporting Page & Engine
  // ==========================================
  function initAutoReporter() {
    if (!appData) return;
    
    const templateSelect = document.getElementById("reporter-template-select");
    const regionSelect = document.getElementById("reporter-region-select");
    
    templateSelect.innerHTML = "";
    regionSelect.innerHTML = "";
    
    appData.report_templates.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.id;
      opt.textContent = t.title;
      templateSelect.appendChild(opt);
    });
    
    appData.regions.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.id;
      opt.textContent = r.name;
      regionSelect.appendChild(opt);
    });
    
    // Generate Report Preview Button
    document.getElementById("btn-generate-report").addEventListener("click", () => {
      const regionId = regionSelect.value;
      const targetRegion = appData.regions.find(r => r.id === regionId);
      const selectedTemplateId = templateSelect.value;
      const targetTemplate = appData.report_templates.find(t => t.id === selectedTemplateId);
      
      generateReportContent(targetRegion, targetTemplate);
    });
    
    // Download File simulation
    document.getElementById("btn-download-report").addEventListener("click", () => {
      const regionId = regionSelect.value;
      const targetRegion = appData.regions.find(r => r.id === regionId);
      const selectedTemplateId = templateSelect.value;
      const targetTemplate = appData.report_templates.find(t => t.id === selectedTemplateId);
      
      const markdownContent = generateReportMarkdown(targetRegion, targetTemplate);
      
      // Trigger browser file download
      const blob = new Blob([markdownContent], { type: "text/markdown;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.setAttribute("download", `R_D_인구소멸대응보고서_${targetRegion.name.split(" ")[1]}.md`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      addConsoleLog(`INFO: 보고서 다운로드 성공 - R_D_인구소멸대응보고서_${targetRegion.name.split(" ")[1]}.md`);
    });
  }

  function generateReportContent(region, template) {
    const previewEl = document.getElementById("report-preview");
    
    // Generate structured preview text with active metadata
    let html = `
      <h2>${template.title}</h2>
      <p style="text-align: right; font-size:12px; color: #4b5563;">보고서 번호: RD-POP-2026-${region.id.toUpperCase()}</p>
      <p style="text-align: right; font-size:12px; color: #4b5563;">발생 일시: 2026년 05월 23일 13:00</p>
      
      <h3>1. 대상 지자체 기본 현황 및 예측 요약</h3>
      <p>
        본 분석서의 대상인 <strong>${region.name}</strong>은 현재 등록 인구수 <strong>${region.population.toLocaleString()}명</strong>, 
        평균 가중 출산율 <strong>${region.birthRate}명</strong>으로 고령화 지수가 <strong>${region.agingIndex}%</strong>에 달해 인구소멸 위험지수 <strong>${region.riskIndex}</strong> 등급의 극심한 소멸 위험 지역입니다.
        AI 예측 모델에 따르면, 현행 유지 시 10년 후 인구는 약 <strong>${Math.round(region.population * 0.81).toLocaleString()}명</strong> 수준으로 급감할 것으로 예측됩니다.
      </p>
      
      <h3>2. MLOps 인공지능 모델 검증지표</h3>
      <ul>
        <li>글로벌 협업 모델 Accuracy: 0.892 (SOTA 기준)</li>
        <li>연합 사일로 데이터 최근 10시간 MSE 오차: 0.041 만족</li>
        <li>입력 데이터 분산 안정성 (PSI): ${driftInjected ? "0.384 (드리프트 위험 발생)" : "0.045 (매우 안정)"}</li>
      </ul>
      
      <h3>3. SHAP 중요 기여 특성에 따른 최적 맞춤 대책</h3>
      <p>
        인구소멸을 지연시키는 데 가장 큰 양의 기여를 하는 인자는 <strong>청년층 복지 재정 (+0.354)</strong> 및 <strong>제조업 일자리 수 (+0.281)</strong>이며, 
        평균 연령 (-0.152)의 증가는 인구 감소를 가속화하는 핵심 위험 요인으로 파악되었습니다.
        따라서 본 지자체는 청년 유입을 극대화할 수 있는 다음과 같은 특화 예산 배정을 제안합니다.
      </p>
      
      <h3>4. 제언 및 최종 권고 요약</h3>
      <ol>
        <li>welfare 예산 부문에 청년 보조 자금 배정 가중치를 최소 60% 이상으로 확대 편성.</li>
        <li>industry 부문의 산업단지 유치를 유도하여 청년 근로자의 유입 세제 혜택 가속화.</li>
        <li>정량적 성과 모니터링 강화를 위해 3차년도 MLOps 대시보드 실시간 API 연계 체계 가동.</li>
      </ol>
      
      <div style="margin-top: 30px; border-top: 1px solid #d1d5db; padding-top: 20px; font-size:12px; color: #6b7280; text-align: center;">
        국토인구소멸대응 공동 R&D 플랫폼 데이터 연계 승인필
      </div>
    `;
    
    previewEl.innerHTML = html;
    addConsoleLog(`INFO: 보고서 미리보기 생성 성공 - ${region.name}`);
  }

  function generateReportMarkdown(region, template) {
    return `# ${template.title}
    
보고서 번호: RD-POP-2026-${region.id.toUpperCase()}
발생 일시: 2026년 05월 23일 13:00

## 1. 분석 개요 및 대상 지자체 기본 현황
본 R&D 보고서는 국토 소멸 대응 프로젝트의 3차년도 연동 모델 결과에 기반하여 지자체 맞춤형 복지 예산과 인구 이동간의 상관관계를 다각 분석한 권고안입니다.

* 대상 지자체: ${region.name}
* 인구수: ${region.population.toLocaleString()}명
* 출산율: ${region.birthRate}명
* 고령화지수: ${region.agingIndex}%
* 인구소멸 위험지수: ${region.riskIndex} (소멸 위험등급 경보)

## 2. MLOps AI 모델 검증지표
* 예측 모델 정확도 (Accuracy): 0.892
* 데이터 분산 안정성 (PSI): ${driftInjected ? "0.384 (드리프트 감지)" : "0.045 (정상)"}
* 검출된 이상치 (Outliers): ${driftInjected ? "3건" : "0건"}

## 3. SHAP 특징 중요도 기여 요인 분석
* 1순위 기여: 청년 복지 예산 가중치 (SHAP: +0.354)
* 2순위 기여: 제조업 공장 일자리 유치 (SHAP: +0.281)
* 3순위 위험 요인: 평균 연령 증가 (SHAP: -0.152)

## 4. 정책 효과 시뮬레이션 예측
* 현 정책 유지 시 10년 후 인구수: ${Math.round(region.population * 0.81).toLocaleString()}명
* 추천 맞춤 정책 (복지 가중치 집중) 적용 시 10년 후 인구수: ${Math.round(region.population * 0.95).toLocaleString()}명

## 5. 지자체 정책 실무 요약 권고 사항
1. 청년 영유아 복지 예산 가중치를 확대하여 정주 요건 조기 개선.
2. 메타데이터 API 규격 스키마를 통해 실시간 데이터 수집 및 drift 감지 모듈의 조기 구축.

---
© 2026 국토인구소멸대응 공동 R&D 통합 플랫폼 R-Center. All rights reserved.
`;
  }

  // ==========================================
  // 10. Test Event Inject Panel Controls
  // ==========================================
  function initInjectControls() {
    const btnNormal = document.getElementById("btn-inject-normal");
    const btnDrift = document.getElementById("btn-inject-drift");
    const quickAlert = document.getElementById("quick-alert-status");
    
    btnNormal.addEventListener("click", () => {
      if (pipelineRunning) {
        alert("파이프라인이 이미 동작 중입니다. 완료 후 초기화해 주세요!");
        return;
      }
      driftInjected = false;
      updateMLOpsMonitorData();
      quickAlert.style.display = "none";
      addConsoleLog("INFO: 정상 시나리오 데이터가 실시간 API에 입력되었습니다.");
    });
    
    btnDrift.addEventListener("click", () => {
      if (pipelineRunning) return;
      driftInjected = true;
      updateMLOpsMonitorData();
      
      // Trigger warning UI popup
      quickAlert.style.display = "block";
      quickAlert.textContent = "데이터 드리프트 감지됨! 슬랙 경고 발송 완료.";
      showDriftWarningPopup();
      
      // Auto-trigger retraining pipeline after 1s
      addConsoleLog("WARN: 데이터 드리프트 심각수준 감지! (PSI: 0.384 > 0.20)", false, true);
      addConsoleLog("INFO: MLOps 오케스트레이터가 재학습 자동 스케줄링을 시작합니다...");
      
      setTimeout(() => {
        startOrchestratorPipeline();
      }, 1500);
    });
  }

  function showDriftWarningPopup() {
    const container = document.getElementById("popup-container");
    const popup = document.createElement("div");
    popup.className = "alert-popup";
    popup.innerHTML = `
      <i class="fa-solid fa-triangle-exclamation" style="font-size:24px; color: var(--accent-red);"></i>
      <div>
        <div style="font-weight:700; color: #ffffff; font-size:14px;">[경보] 데이터 드리프트 발생</div>
        <div style="font-size:11px; color: var(--text-secondary); margin-top:2px;">
          실시간 수집 분포 불안정 (PSI: 0.384). MLOps 오케스트레이터 가동.
        </div>
      </div>
      <button class="alert-icon-btn btn-danger" style="width:24px; height:24px; font-size:10px; margin-left:12px;" onclick="this.parentElement.remove()">X</button>
    `;
    container.appendChild(popup);
    
    // Auto remove after 5s
    setTimeout(() => {
      popup.remove();
    }, 5000);
  }

  // ==========================================
  // 11. Start Loader
  // ==========================================
  await loadData();
});
