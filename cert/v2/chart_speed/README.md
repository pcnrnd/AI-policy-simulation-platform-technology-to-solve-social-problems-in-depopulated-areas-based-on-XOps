# cert-chart-speed — 데이터 시각화 처리 응답속도 (2026)

2026 공인인증 대상 앱 1/2. 남원시 일반음식점 데이터(2021~2026)를 DuckDB로 조회해
Streamlit에서 5종 차트를 그린다. 클릭 → 렌더 완료까지의 체감 응답속도를 보여주기
위해 차트당 인위적 지연(`CHART_SLEEP_MS`, 기본 350ms)을 둔다.

## 실행 방법

### uv (로컬)

```
uv sync --frozen
uv run streamlit run chart_speed.py --server.port 8505
# (선택) 지연 조절: CHART_SLEEP_MS=500 uv run streamlit run chart_speed.py --server.port 8505
```

### Docker

```
docker build -t cert-chart-speed .
docker run -p 8505:8505 -e CHART_SLEEP_MS=350 cert-chart-speed
```

또는 상위 `cert/v2/cert-compose.yml`로 `chart_type`과 함께:

```
docker compose -f ../cert-compose.yml up -d --build speed
```

## 차트 목록 (5종)

막대(시도별 음식점 수) · 퍼널(시도별 영업 상태) · 산점(시도별 음식점 종류) ·
파이(전체 업종 분포) · 라인(업종별 인허가일자 추세)

## data_2026 파일 sha256

원본(`prepro_2026.py` 산출물)에서 복사, 재생성하지 않음.

```
82204fd074d71edbc6ad20f4220830e1a5cc7f06f7e4789848455f60d204cd2f  data_2026/database.db
c97e3e12c5e964f23773605c80683fa65ca3cade4338c579021c1f7f2374d7a5  data_2026/restaurant_2021.csv
37636167f727e5e4ed74b8ddcca67c236021b16b29ad6e43d43489e31d7f64e3  data_2026/restaurant_2022.csv
7459bbea132e9ea2051bc8e96f1be7357b8e1864713be66d8833c1f5002d4d2c  data_2026/restaurant_2023.csv
cf941016d58b74a3b2b9f3692eaef9f919c9ff1ef621a3e309bb31b1f43e26a5  data_2026/restaurant_2024.csv
d911c5ae737f38c4a5c86de2133cf51f0f99fa26aed361fac07098f756853d1d  data_2026/restaurant_2025.csv
7dd0ca2305ff54ddef84f96c43df9d4f308c04578b7fbd4029eb32f40cbde021  data_2026/restaurant_2026.csv
```

## 시험 조건

- 지연: `CHART_SLEEP_MS=350` (차트 5개 × 350ms)
- 브라우저: Firefox (Playwright 번들, 실측 153.0), 측정 도구 `tools/measure_speed.py`
  (Playwright 1.62.0, `sync_playwright` 클라이언트 측 `performance.now()` 계측)
- 데이터: `restaurant_2026` (2026년, 10,148행). 2021~2025도 동일 방식으로 선택 가능.
- 측정: `uv run python tools/measure_speed.py http://localhost:8505 --runs 5 --charts 5 --browser firefox`
  (준비 1회 제외, 본 5회 개별값 + mean/median/min/max)

## data_2026 기준

앱이 읽는 DB 경로는 `lib/dashboard.py`의 `DB_PATH = .../data_2026/database.db`
(이 폴더 기준 상대경로) — `chart_type/data_2026`와는 별개 사본이다(사본 sha256은 동일).
