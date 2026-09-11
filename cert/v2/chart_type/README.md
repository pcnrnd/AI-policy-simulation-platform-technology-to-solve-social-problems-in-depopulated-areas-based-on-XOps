# cert-chart-type — 데이터 시각화 처리 유형 (2026)

2026 공인인증 대상 앱 2/2. 같은 남원시 일반음식점 데이터(2021~2026)를 DuckDB로
조회해 Streamlit에서 8종 차트를 그린다. 응답속도용 인위적 지연은 없다.

## 실행 방법

### uv (로컬)

```
uv sync --frozen
uv run streamlit run chart_type.py --server.port 8506
```

### Docker

```
docker build -t cert-chart-type .
docker run -p 8506:8506 cert-chart-type
```

또는 상위 `cert/v2/cert-compose.yml`로 `chart_speed`와 함께:

```
docker compose -f ../cert-compose.yml up -d --build types
```

## 차트 목록 (8종)

막대(시도별 음식점 수) · 퍼널(시도별 영업 상태) · 산점(시도별 음식점 종류) ·
파이(전체 업종 분포) · 라인(업종별 인허가일자 추세) ·
히트맵(시도·업종 집중도) · 트리맵(시도별 업종 규모) · 생키(시도·영업상태·업종 흐름)

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

- 지연: 없음(`CHART_SLEEP_MS` 미적용)
- 브라우저: Firefox (Playwright 번들, 실측 153.0). 이 앱 전용 측정 스크립트는 없고,
  `chart_speed/tools/measure_speed.py <url> --charts 8`로 렌더 완료를 확인할 수 있다.
- 데이터: `restaurant_2026` (2026년, 10,148행). 2021~2025도 동일 방식으로 선택 가능.

## data_2026 기준

앱이 읽는 DB 경로는 `lib/dashboard.py`의 `DB_PATH = .../data_2026/database.db`
(이 폴더 기준 상대경로) — `chart_speed/data_2026`와는 별개 사본이다(사본 sha256은 동일).

## 참고: 데이터 점검 노트북

`../tools/data_check_2026.ipynb`가 이 폴더의 `data_2026/database.db`를 기준으로
8종 차트 집계 쿼리를 점검한다(경로: `../chart_type/data_2026/database.db`).
