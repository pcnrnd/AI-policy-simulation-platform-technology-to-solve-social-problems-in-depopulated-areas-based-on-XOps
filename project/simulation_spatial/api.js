// 읽기 전용 로컬 API — 고른 시군구들(/api/data?regions=52190,52130, 없으면 REGION_CODE)의 시설·행정동 인구·이름·경계를 화면에 준다.
// 접속 정보는 환경변수로만 받는다(DB_USER·DB_PASSWORD·REGION_CODE 필수, DB_HOST·DB_PORT·DB_NAME 선택). 접속 정보는 어디에도 출력하지 않는다.
// - DB(flows, 읽기 전용): 노인 돌봄 시설(ingu_fclty_oldman)만 읽는다.
// - 행정동 경계: BOUNDARY_FILE(없으면 data/의 가장 최신 HangJeongDong_ver*.geojson). 시작할 때 전국을 읽어 시군구별로 나눠 둔다.
// - 인구: POP_FILE(없으면 data/pop/의 가장 최신 "…주민등록 인구수_*.csv", 행정안전부 행정동 성별·연령별 주민등록 인구). 시작할 때 한 번 읽는다.
// - 의료·보건 시설: HIRA_DIR(없으면 data/hira/)의 건강보험심사평가원 병의원·약국·진료과목 CSV. 첫 요청 때 한 번 읽고 지역별 결과는 따로 둔다.
// 시설은 좌표로 그 지역 행정동 경계 안에 넣어 읍면동(행정동 10자리)을 정한다.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import mariadb from "mariadb";
import { boundariesByRegion, boundaryAnchors, boundaryOfPoint } from "./src/lib/boundary.js";
import { csvRows, healthFacilities, inRegion, readCsv } from "./src/lib/hira.js";
import { oldRegionCode, toFacilities } from "./src/lib/ingu.js";
import { aggregateRows, mergeOrphans, populationFor, regionList } from "./src/lib/population.js";

const { DB_USER, DB_PASSWORD, REGION_CODE } = process.env;
if (!DB_USER || !DB_PASSWORD) {
  console.error("DB 접속 정보가 없습니다. DB_USER와 DB_PASSWORD 환경변수를 넣고 다시 실행하세요.");
  process.exit(1);
}
if (!/^\d{5}$/.test(REGION_CODE ?? "")) {
  console.error("지역 코드가 없습니다. REGION_CODE 환경변수(기본 시군구 코드 5자리)를 넣고 다시 실행하세요.");
  process.exit(1);
}
// 한 번에 고를 수 있는 시군구 수 상한(화면이 느려지지 않게)
const MAX_REGIONS = 8;

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "data");
const decode = (file) => new TextDecoder("euc-kr").decode(fs.readFileSync(file));

/** dir 안에서 이름이 pattern에 맞는 가장 마지막(최신) 파일. 없으면 null. */
function latestFile(dir, pattern) {
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => pattern.test(f)).sort() : [];
  return files.length ? path.join(dir, files[files.length - 1]) : null;
}

/** 전국 행정동 경계를 시군구별로. 파일이 없으면 빈 Map(경계 없이 동작). */
function loadBoundaries() {
  const file = process.env.BOUNDARY_FILE || latestFile(DATA_DIR, /^HangJeongDong_ver\d+\.geojson$/);
  if (!file || !fs.existsSync(file)) {
    console.log("경계 파일이 없어 경계 없이 실행합니다.");
    return new Map();
  }
  const map = boundariesByRegion(JSON.parse(fs.readFileSync(file, "utf8")));
  console.log(`경계 ${[...map.values()].reduce((s, l) => s + l.length, 0)}곳·시군구 ${map.size} (${path.basename(file)})`);
  return map;
}
const BOUNDARIES = loadBoundaries();
const ALL_BOUNDARIES = [...BOUNDARIES.values()].flat();
const ANCHORS = boundaryAnchors(ALL_BOUNDARIES);

/** 행정동 인구: CSV를 행정동별 구간으로 모으고, 경계 없는 출장소 행은 이름이 앞부분인 경계(부모 읍면동)에 합친다. */
function loadPopulation() {
  const file = process.env.POP_FILE || latestFile(path.join(DATA_DIR, "pop"), /주민등록\s*인구수.*\.csv$/);
  if (!file || !fs.existsSync(file)) {
    console.error("인구 파일이 없습니다. data/pop/에 행정동 성별·연령별 주민등록 인구 CSV를 두고 다시 실행하세요.");
    process.exit(1);
  }
  const { header, rows } = csvRows(decode(file));
  const pop = aggregateRows(header, rows);
  const { dongs, merged, unmatched } = ALL_BOUNDARIES.length ? mergeOrphans(pop.dongs, ALL_BOUNDARIES) : { dongs: pop.dongs, merged: [], unmatched: [] };
  console.log(`인구 ${pop.month} 행정동 ${Object.keys(pop.dongs).length}곳(출장소 합침 ${merged.length}, 경계 없음 ${unmatched.length}) (${path.basename(file)})`);
  return { month: pop.month, dongs };
}
const POP = loadPopulation();
const REGIONS = regionList(POP.dongs);
const REGION_BY_CODE = Object.fromEntries(REGIONS.map((r) => [r.code, r]));
if (!REGION_BY_CODE[REGION_CODE]) {
  console.error(`기본 지역 코드 ${REGION_CODE}가 인구 파일에 없습니다.`);
  process.exit(1);
}

/** HIRA 병의원·약국(필요한 열만)과 산부인과 진료과목 행. 파일이 하나라도 없으면 null. 첫 요청 때 한 번 읽는다. */
let HIRA;
function hiraRows() {
  if (HIRA !== undefined) return HIRA;
  const dir = process.env.HIRA_DIR || path.join(DATA_DIR, "hira");
  const files = { hospitals: /병원정보서비스.*\.csv$/, pharmacies: /약국정보서비스.*\.csv$/, departments: /진료과목정보.*\.csv$/ };
  const paths = Object.fromEntries(Object.entries(files).map(([k, p]) => [k, latestFile(dir, p)]));
  if (Object.values(paths).some((p) => !p)) {
    console.log("병의원·약국 파일이 없어 의료·보건 시설 없이 실행합니다.");
    return (HIRA = null);
  }
  const cols = ["암호화요양기호", "종별코드명", "시도코드명", "시군구코드명", "좌표(X)", "좌표(Y)"];
  const slim = (file) => readCsv(decode(file)).map((r) => Object.fromEntries(cols.map((c) => [c, r[c]])));
  HIRA = {
    hospitals: slim(paths.hospitals),
    pharmacies: slim(paths.pharmacies),
    departments: readCsv(decode(paths.departments), (r) => r.진료과목코드명 === "산부인과").map((r) => ({ 암호화요양기호: r.암호화요양기호, 진료과목코드명: r.진료과목코드명 }))
  };
  console.log(`병의원 ${HIRA.hospitals.length}·약국 ${HIRA.pharmacies.length}·산부인과 진료과목 ${HIRA.departments.length}행`);
  return HIRA;
}

/** 한 시군구의 의료·보건 시설(지역 시도·시군구 이름으로 거름). 지역별로 한 번만 만든다. */
const HEALTH = new Map();
function healthFor(code) {
  if (HEALTH.has(code)) return HEALTH.get(code);
  const all = hiraRows();
  const r = REGION_BY_CODE[code];
  let list = [];
  if (all && r) {
    const keep = inRegion({ sido: r.sido, gugun: r.name });
    const hospitals = all.hospitals.filter(keep);
    const ids = new Set(hospitals.map((h) => h.암호화요양기호));
    list = healthFacilities({ hospitals, pharmacies: all.pharmacies.filter(keep), departments: all.departments.filter((d) => ids.has(d.암호화요양기호)) });
    console.log(`의료·보건 ${code} ${r.sido} ${r.name}: ${list.length}곳`);
  }
  HEALTH.set(code, list);
  return list;
}

const pool = mariadb.createPool({
  host: process.env.DB_HOST || "127.0.0.1",
  port: Number(process.env.DB_PORT || 3310),
  database: process.env.DB_NAME || "flows",
  user: DB_USER,
  password: DB_PASSWORD,
  connectionLimit: 2,
  initSql: "SET SESSION TRANSACTION READ ONLY"
});

/** 요청의 regions 값 → 인구 파일에 있는 시군구 코드(중복 없이, 최대 MAX_REGIONS). 없으면 REGION_CODE 하나. */
function parseRegions(param) {
  const codes = [...new Set(String(param ?? REGION_CODE).split(",").map((c) => c.trim()).filter((c) => /^\d{5}$/.test(c) && REGION_BY_CODE[c]))];
  return codes.slice(0, MAX_REGIONS);
}

async function loadData(codes) {
  const conn = await pool.getConnection();
  try {
    // 시설 표는 옛 코드(전북 45·강원 42)와 새 코드가 섞여 있어 둘 다 찾는다
    const likes = [...new Set(codes.flatMap((c) => [c, oldRegionCode(c)]))].map((c) => `${c}%`);
    const facilityRows = await conn.query(
      `SELECT fclty_ty, map_x, map_y, addr10 FROM ingu_fclty_oldman WHERE ${likes.map(() => "addr10 LIKE ?").join(" OR ")} ORDER BY fclty_ty, addr10, map_x, map_y`,
      likes
    );
    const boundaries = codes.flatMap((c) => BOUNDARIES.get(c) ?? []);
    const dongOf = boundaryOfPoint(boundaries);
    const facilities = [...toFacilities(facilityRows), ...codes.flatMap(healthFor)].map((f) => ({ ...f, dong: dongOf(f) }));
    const { population, names } = populationFor(POP, codes, ANCHORS);
    return { region: codes.join(","), regions: codes, regionList: REGIONS, facilities, names, population, boundaries };
  } finally {
    conn.release();
  }
}

const send = (res, status, body) => {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (req.method !== "GET" || url.pathname !== "/api/data") return send(res, 404, { error: "없는 주소" });
  const codes = parseRegions(url.searchParams.get("regions") ?? undefined);
  if (!codes.length) return send(res, 400, { error: "지역 코드가 맞지 않습니다" });
  try {
    send(res, 200, await loadData(codes));
  } catch (err) {
    console.error(`DB 조회 실패: ${err.code ?? err.message}`); // 오류 코드만 남긴다(접속 정보 미포함)
    send(res, 502, { error: "DB 조회 실패" });
  }
});

const port = Number(process.env.API_PORT || 5182);
server.listen(port, "127.0.0.1", () => console.log(`API http://127.0.0.1:${port}/api/data?regions=코드,코드 (기본 지역 ${REGION_CODE}, 지역 ${REGIONS.length}개)`));
