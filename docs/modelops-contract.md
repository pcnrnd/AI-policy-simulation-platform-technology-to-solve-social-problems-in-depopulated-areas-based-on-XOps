# 모델 운영 계약

모델 종류는 요인분석(`factor_analysis`)과 시뮬레이션(`simulation`) 두 개다. 지역·target별 인스턴스는 각 종류 안에서 관리하며 버전은 등록 데이터셋과 학습 job에 고정된다. 기존 실측·시드 활성 버전과 이전 이력은 삭제하거나 전환하지 않는다.

앱은 모든 API에 `X-Data-Mode: mock|real`을 전달한다. `/modelops`는 미지정 시 real이며, 레거시 API는 헤더가 없으면 기존 동작을 유지한다. 명시 모드가 있으면 include_seed보다 모드가 우선한다. mock은 seed 입력만 사용하고 real은 등록한 실제 데이터만 사용한다. 모드가 다른 ID는 단건 조회·실행·활성화·복원·삭제에도 사용할 수 없다. `data:read`와 `data:write` 권한은 기존 토큰 계약을 따른다.

## 등록과 학습

1. `GET /api/v3/modelops/available-datasets`에서 현재 모드의 원본 ID/hash·지역·target·기간을 확인한다.
2. `POST /modelops/datasets`에 name, source_dataset_id, source_hash를 전달한다. 서버는 원본을 수정하지 않고 private 등록본을 생성한다. 등록 결과는 dataset_id, content_hash, source_dataset_id, mode/source, region_id, target, observed_from/to, features, row_count, private_hash와 재현 가능한 저장 경로를 포함한다.
3. `GET /modelops/datasets`로 등록 결과를 재조회한다. `POST /modelops/pipelines`는 id, name, model_type, dataset_id, dataset_hash와 trigger_policy를 받는다. 서버에서 hash·모드·원본 참조를 확인해 인스턴스를 연결한다. 같은 모드/종류/지역/target 인스턴스는 여러 등록본 버전을 가질 수 있다.
4. `POST /modelops/pipelines/{id}/run`으로 실제 학습을 접수한다. `GET /modelops/jobs/{id}`에서 queued/running → saved/failed를 확인한다. saved는 버전·아티팩트·감사 기록이 저장됐다는 뜻이며 활성화를 뜻하지 않는다. 활성화는 별도 요청이다.
5. `POST /modelops/instances/{id}/versions/{version}/activate`는 완료 job·dataset hash·아티팩트 hash와 기준선보다 개선된 MAE를 확인한다. `rollback`은 해당 인스턴스의 이전 활성 버전에만 적용한다. 기존 레거시 활성 모델은 변경하지 않는다.

새 기록은 모드+종류+ID로 분리한 `mo_assets`에 추가한다. private 학습 입력과 아티팩트는 app model storage 아래 modelops/mock 또는 modelops/real 경로에 저장하며 Git 추적 대상이 아니다. real 학습은 등록본의 관측행과 기존 Ridge·순차 평가 함수를 사용하고, mock 학습은 seed 지역 시계열과 교차 검증을 사용한다. 실패를 다른 모드의 수치로 대체하지 않는다.

## Phase C에서 사용할 버전 인터페이스

`GET /modelops/instances/{id}/versions`는 버전 id, instance_id, model_type, mode/source, region_id, target, dataset_id/hash, source_dataset_id, observed_from/to, features, job_id, metrics/baseline, artifact_path/hash, created_at, status(active/previous/candidate)를 반환한다.

`GET /modelops/instances/{id}/versions/{version}/artifact`는 등록본·아티팩트 hash를 검증한 뒤 다음 값을 반환한다.

| 필드 | 의미 |
|---|---|
| features, coef | 동일 순서의 입력 피처와 표준화 공간 계수 |
| intercept | 기준값 |
| train_means, train_stds | 동일 순서의 학습 정규화 통계 |
| model_type, model_id, version | 모델 종류·인스턴스·고유 버전 |
| mode/source, region_id, target | 입력 범위 |
| dataset_id, content_hash, dataset_path, job_id | 불변 등록본·내용 해시·학습 실행 |

예측값은 `intercept + Σ coef[i] × (x[i]-train_means[i])/train_stds[i]`이고 같은 각 항을 요인 기여도로 사용할 수 있다. 이는 모델의 조건 입력 예측/선형 기여도이며 정책 변경의 인과 효과를 뜻하지 않는다. real 버전은 원 관측기간·평가기간·예측월도 보존한다. 추론→시뮬레이션→추천 UI 연결은 Phase C에서 수행한다.

## 활성 버전 추론과 관측 입력 민감도 (Phase C)

`GET /modelops/instances/{id}/versions/{version}/inputs?dong_code=`는 활성 버전만 사용하며 등록본 private hash·모델 artifact hash·지역/target/종류·job 연결을 재검증한다. 실제 입력값, 행정동 목록, 조정 가능한 연속 관측값의 학습 범위, model/dataset/version/job/평가기간/예측기간 근거를 반환한다. real 입력은 등록 자료의 마지막 관측 다음 달이며, 과거 관측값의 lag·전년동월·계절/지역 인코딩으로 만든다. 범위는 **선택 행정동의 학습 구간**(`range_provenance` from/to/row_count)으로 계산한다. mock API 입력은 private seed의 마지막 관측 기간에 대한 기존 학습 입력이며 미래 예측이라고 표시하지 않는다.

`POST .../predict`는 `region_id`, `target`, `dataset_id`, `dataset_hash`, 선택 `dong_code` 및 `adjustments`(조정 가능한 feature → 유한 수치)를 받는다. 활성 버전이 바뀌거나 입력 근거가 다르면 409, 조정 불가/비유한/누락 입력은 422, 다른 모드 ID는 404다. 모든 계산은 read 권한으로 수행하며 저장·학습·활성 변경은 없다. 응답의 `inference_id`는 영속 기록이 아니라 해당 계산 결과의 연결 식별자다.

반환 `prediction`은 artifact 수식으로 계산하고 `intercept + Σ contributions.value`와 일치한다. `baseline_inputs`/`baseline_prediction` 대비 조정 `inputs`/`prediction`의 `delta`, 기여도의 입력/계수/평균/표준편차와 실제 값, 관측 범위 밖 입력 목록을 함께 반환한다. 시뮬레이션 응답 `recommendations`는 각 조정 가능한 관측 입력을 선택 행정동 학습 범위의 1%만큼 증가/감소한 단일 변경 시나리오의 실제 prediction/delta다. 전년 관측이 없으면 기존 feature 규칙대로 전월 값의 파생 대체 입력을 함께 갱신한다.

real UI는 두 활성 모델의 지역·target·원 내용 hash·입력 좌표/기간·feature가 같을 때 연결하며 등록본 ID가 다르면 각 ID를 명시한다. 기여도 → 입력 가정 예측 → **예측 민감도 기반 검토 우선순위**로 표시한다. 현재 관측 lag에는 정책 변수가 없으므로 정책 효과나 RICE 점수를 제시하지 않는다. 실제 외생 정책 입력을 가진 등록·학습 스키마는 후속 범위다. 활성/대응 자료가 없으면 등록→학습→활성화 안내만 제공한다.

실데이터 진단 UI는 제거했고 `GET /realdata/analysis`는 인증 유지·deprecated·410으로 퇴역했다. 진단 계산 모듈/원 기록과 모니터의 별도 evaluation/explain/drift 평가 패널은 보존한다.

## 모드 전환과 외부 모듈

전환은 진행 중 앱 요청을 취소하고 응답 epoch·토큰 캐시를 무효화하며 선택·폴링·알림·지표 override·리포트 미리보기를 정리한다. 추천 시나리오 저장 키도 모드별로 분리한다. 서버의 진행 중 작업과 감사 기록은 보존한다.

현재 대시보드와 리포팅은 자체 모드 전달/데이터 표시만 처리하며 새 modelops와 결합하지 않는다. 외부 upstream이 제공되기 전에는 연동 완료로 주장하지 않는다. 데모 리포트는 실측 API를 호출하지 않으며 real 리포트에는 실측이 없는 시드 인구/정책 수치를 포함하지 않는다. HTML 출력은 HTML(.html)로 표시하고 native HWP 선택지는 제공하지 않는다.

실행 ID와 학습 시도별 아티팩트 ID는 UUID로 발급한다. 레거시 논리 모델 버전 표기는 유지하지만 시도 파일은 서로 다른 경로를 사용한다. 이전에 발생한 중복 persisted ID/파일 경로 이력은 원본 보존 원칙에 따라 재작성하지 않는다.
