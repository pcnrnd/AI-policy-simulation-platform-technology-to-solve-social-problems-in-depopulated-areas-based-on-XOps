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
| `namwon_bccard_consumption_240130.xlsx` | BC카드 남원시 지역소비데이터(행정동×업종×성별) | 2시트(27 / 212,343행) | 2026-09-10 |
| `gwto_kt_tourism_indicators_202512.xlsx` | KT GWTO 관광 분석 지표 총괄(2025년 12월) | 26시트 | 2026-09-10 |
| `namwon_kt_visitors_20231130.xlsx` | 남원시 행정동 KT 방문객 데이터 | 2시트(1,335 / 21,343행) | 2026-09-10 |
| `nowon_kt_festival_20250919.xlsx` | 노원구 축제지 KT 방문객 데이터 | 11시트 | 2026-09-10 |

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

### 외부데이터 — BC카드·KT·GWTO (수집일 2026-09-10)

원본은 xlsx 그대로 바이트 무변경 복사본이다(파일명만 영문화). 원본 위치는
`F:\pcn_onedrive\OneDrive - 피씨엔\문서\01_연구과제업무\2026\인구감소\data\20260910_외부데이터_BC카드남원소비_남원행정동_노원축제지_KT관광\`
이다. sha256 앞 6자리는 카탈로그 lineage.commit 규약을 따른다.

- **`namwon_bccard_consumption_240130.xlsx`**
  원본 파일명: `BC카드_남원시 지역소비데이터_1.행정동X업종X성별_240130_kwA3J7gr.xlsx`
  제공기관: BC카드. sha256 앞 6자리 `89a950`.
  비식별 조건: k-anonymity(k≥3) 적용으로 일부 셀 값 누락 존재.
  데이터 기간: 2019·2022·2023년 각 12개월(총 36개월).
  이용 조건: 협약·계약 조건 확인 필요 [미확인].

- **`gwto_kt_tourism_indicators_202512.xlsx`**
  원본 파일명: `KT_GWTO_관광 분석 지표_총괄_25년 12월(1).xlsx`
  제공기관: 강원관광재단 GWTO 보고서(KT 데이터). sha256 앞 6자리 `1c5a35`.
  데이터 기간: 2020.01~2025.12.
  이용 조건: 협약·계약 조건 확인 필요 [미확인].

- **`namwon_kt_visitors_20231130.xlsx`**
  원본 파일명: `남원시_행정동_납품용_20231130.xlsx`
  제공기관: KT. sha256 앞 6자리 `bd6d3a`.
  데이터 기간: 201901~202310.
  이용 조건: 협약·계약 조건 확인 필요 [미확인].

- **`nowon_kt_festival_20250919.xlsx`**
  원본 파일명: `노원구_축제지(1)_납품용_20250919.xlsx`
  제공기관: KT. sha256 앞 6자리 `5e10cf`.
  데이터 기간: 20250814~20250915.
  이용 조건: 협약·계약 조건 확인 필요 [미확인].

위 4개 xlsx(약 12MB)는 git에 추적하지 않는다(2026-09-11 사용자 결정, `.gitignore`의 `platform/data/real/*.xlsx`). 원본 정본은 위 OneDrive 경로이며, 재현은 xlsx를 이 폴더에 다시 복사한 뒤 `extract_sheets.py` → `prepare_sheets.py` → `load_external_data.py` 순으로 실행한다.

## 원칙

- **전국 보존**: 경계 데이터는 남원·신안만 추리지 않고 전국 251개를 그대로 적재한다
  (지자체 간 벤치마크용). 복지시설은 지자체 제공 파일 특성상 남원·신안 분량이다.
- **실 코드 사용**: `tb_welfare_facility.region_code` 는 실제 법정 시군구 코드
  (남원 45190, 신안 46910)를 쓴다. 기존 합성 시드가 쓰는 46900(비실재 코드)과 다르다.
- 카탈로그 lineage.commit 은 각 데이터 파일 sha256 앞 6자리다
  (경계 `8c406e`, 복지시설 남원 `189bc9`·신안 `e9afef`).

## 적재 방법

```powershell
# DB 기동(WSL): docker compose -f platform/compose.xops.yaml up -d pg
pip install "psycopg[binary]"
python platform/scripts/load_real_data.py            # 기본 DSN: localhost:5433/xops_dataops
python platform/scripts/load_real_data.py --pg-dsn postgresql://xops:xops@localhost:5433/xops_dataops
# BC카드·KT·GWTO 외부데이터(prepared CSV 39건). xops-service의 safety.py가 점(.) 없는
# 단일 식별자만 허용하므로 별도 스키마를 쓰지 않고 public 스키마에 접두 합성 테이블명으로 적재한다:
# <데이터셋 접두>_<slug>, 예) ext_bccard_dong_industry_sales, ext_kt_namwon_monthly_dong_visitors.
python platform/scripts/load_external_data.py
```
