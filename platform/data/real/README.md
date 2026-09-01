# 실데이터 원본 (DataOps 실 저장소 적재용)

xops-service 실 저장소(PostGIS/PostgreSQL)에 적재되는 **실제 공공데이터** 원본이다.
적재는 `platform/scripts/load_real_data.py` 가 수행하고, 카탈로그 등록은
`platform/frontend/src/assets/mock_data.json` 의 `metadata_schemas`(ds_08·ds_09)에 있다.

## 파일 목록

| 파일 | 내용 | 행수 | 수집일 |
|---|---|---|---|
| `admin_boundary_kostat2013.geojson` | 전국 시군구 행정구역경계 251개 (통계청 2013 기준, 단순화본) | 251 feature | 2026-09-01 |
| `namwon_welfare_facilities_20251107.csv` | 전북특별자치도 남원시 사회복지시설현황 | 11 | 2026-09-01 |
| `sinan_disabled_welfare_facilities_20260608.csv` | 전남 신안군 장애인복지시설현황 (위경도 포함) | 6 | 2026-09-01 |

## 출처

- **행정구역경계**: 통계청(KOSTAT) 행정구역경계 2013을 GeoJSON으로 변환·재배포한
  공개 저장소 [southkorea/southkorea-maps](https://github.com/southkorea/southkorea-maps)
  (`kostat/2013/json/skorea_municipalities_geo_simple.json`). 좌표계 EPSG:4326.
  속성 `code` 는 통계청 행정구역분류 코드(예: 남원시 35050, 신안군 36480)로,
  법정 시군구 코드(남원 45190, 신안 46910)와 **다른 코드 체계**다.
- **남원시 사회복지시설현황**: [공공데이터포털 15094099](https://www.data.go.kr/data/15094099/fileData.do)
  (기준일 2025-11-07, 제공기관 전북특별자치도 남원시)
- **신안군 장애인복지시설현황**: [공공데이터포털 15114967](https://www.data.go.kr/data/15114967/fileData.do)
  (기준일 2026-06-08, 제공기관 신안군)

CSV 2건은 원본 CP949를 UTF-8로 변환한 것 외에 내용 수정이 없다.
공공데이터포털 파일데이터는 공공누리 제1유형(출처표시) 기준으로 이용한다.

## 원칙

- **전국 보존**: 경계 데이터는 남원·신안만 추리지 않고 전국 251개를 그대로 적재한다
  (지자체 간 벤치마크용). 복지시설은 지자체 제공 파일 특성상 남원·신안 분량이다.
- **실 코드 사용**: `tb_welfare_facility.region_code` 는 실제 법정 시군구 코드
  (남원 45190, 신안 46910)를 쓴다. 기존 합성 시드가 쓰는 46900(비실재 코드)과 다르다.
- 카탈로그 lineage.commit 은 각 데이터 파일 sha256 앞 6자리다
  (경계 `8c406e`, 복지시설 `189bc9`).

## 적재 방법

```powershell
# DB 기동(WSL): docker compose -f platform/compose.xops.yaml up -d pg
pip install "psycopg[binary]"
python platform/scripts/load_real_data.py            # 기본 DSN: localhost:5433/xops_dataops
python platform/scripts/load_real_data.py --pg-dsn postgresql://xops:xops@localhost:5433/xops_dataops
```
