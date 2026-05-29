# CSS_rating2 — AI 기반 신용평가시스템 개발 계획

## 0. 프로젝트 개요
개인의 신용정보를 입력받아 부실가능성(default probability)을 0~1 점수로 산출하는 **웹앱 기반 신용평가시스템**.
- 베이스라인: 로지스틱 회귀(전통적 신용평가모델 대용)
- 챔피언: XGBoost (즉시 학습/배포 가능)
- 도전 모델: sLLM (< 10B parameters), GCP에서 QLoRA 파인튜닝

## 1. 데이터셋
**Lending Club 공개 데이터** (`data_source/loan.csv`, 2,260,668 rows × 145 cols, ~1.2GB)

### 라벨 정의
| `loan_status` | 라벨 | 비고 |
|---|---|---|
| Fully Paid | 0 (정상상환) | |
| Charged Off, Default, Late (31-120 days) | 1 (부실) | |
| Late (16-30 days) | 1 (부실, 보수적) | |
| Current, In Grace Period | **제외** | 결과 미확정 |
| Does not meet credit policy | 제외 | 정책 위반 |

### 데이터 누수(Leakage) 방지 — 반드시 제거
대출 발생 *후* 정보는 신청 시점 점수에 쓸 수 없으므로 제거:
- `total_pymnt*`, `total_rec_*`, `recoveries`, `collection_recovery_fee`
- `last_pymnt_*`, `out_prncp*`, `last_fico_*`, `last_credit_pull_d`
- `grade`, `sub_grade`, `int_rate` — Lending Club 자체 신용등급 (모델이 학습할 대상이므로 입력에서 제외하거나 별도 표시)

### 사용 피처 (신청 시점 가용 정보)
인구통계: `addr_state`, `home_ownership`, `emp_length`, `annual_inc`, `verification_status`
대출 신청: `loan_amnt`, `term`, `purpose`, `installment`
신용 이력: `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `earliest_cr_line` → 신용 이력 연수로 변환
FICO: `fico_range_low`, `fico_range_high` (신청 시점 평균값)

## 2. 시스템 아키텍처

```
data_source/loan.csv (Raw)
        │
        ▼
[src/data/preprocess.py]
  ├─ schema 검증
  ├─ leakage 컬럼 제거
  ├─ 결측치/이상치 처리
  ├─ 범주형 인코딩
  └─ stratified train/val/test 분할 (70/15/15)
        │
        ▼
data/processed/{train,val,test}.parquet
        │
        ├─────────────────────────┐
        ▼                         ▼
[src/models/baseline.py]   [src/training/llm_finetune.py]
  XGBoost + Logistic        Qwen2.5-7B + QLoRA
  로컬 학습 가능              GCP A100/L4 필요
        │                         │
        ▼                         ▼
   model.pkl              adapter_model/
        │                         │
        └────────┬────────────────┘
                 ▼
        [src/web/app.py — FastAPI]
          POST /api/score → { default_prob, score, risk_grade }
          GET  /api/compare → 모델 비교 메트릭
                 │
                 ▼
        [frontend/templates/*.html]
          입력 폼 + 결과 카드 + 비교 페이지
```

## 3. 모델 선정 (sLLM ≤ 10B)

| 모델 | Params | 라이선스 | 추천 사유 |
|---|---|---|---|
| **Qwen2.5-7B** | 7B | Apache 2.0 | 한국어/영어 균형, 수치 추론 우수 |
| Llama-3.1-8B | 8B | Llama Community | 영어 강함, 생태계 풍부 |
| Mistral-7B-v0.3 | 7B | Apache 2.0 | 빠른 추론 |
| Gemma-2-9B | 9B | Gemma | 구글 백본, GCP 친화적 |

**1차 선택: Qwen2.5-7B-Instruct** (한국어 사용자 입력 대응 + 무제한 상업 라이선스)

### 학습 전략
- **QLoRA** (4-bit quantization + LoRA rank=16)
- 입력 직렬화: 표 데이터 → 자연어 프롬프트
  ```
  신청자: 연소득 65000달러, 거주형태 임대, 근무 5년,
  대출목적 부채통합, 신청금액 12000달러, 36개월,
  DTI 18.5, FICO 705~709, ...
  → 부실확률: ?
  ```
- 출력: classification head 또는 토큰 확률 (`정상`/`부실`)
- 평가 메트릭: **AUC**, KS statistic

### 성능 목표
**"전통적 모델보다 10% 우수"의 조작적 정의:**
- 베이스라인: 로지스틱 회귀
- 목표: **AUC 절대값 +0.05 이상 개선** (예: 0.70 → 0.75) 또는 KS +0.05
- 측정: hold-out test set (15%)

## 4. 웹앱 (FastAPI + 정적 프론트엔드)

### 엔드포인트
| Method | Path | 기능 |
|---|---|---|
| GET | `/` | 메인 입력 폼 |
| POST | `/api/score` | 부실확률 산출 |
| GET | `/api/compare` | 모델별 성능 비교 JSON |
| GET | `/compare` | 비교 시각화 페이지 |
| GET | `/healthz` | 헬스체크 (Cloud Run용) |

### UI 페이지
1. **메인**: 한국어 입력 폼 — 소득, 대출금액, 신용이력, FICO 등
2. **결과**: 부실확률(%), 신용점수(300~850 환산), 등급(A~E), 주요 위험 요소
3. **비교**: 로지스틱 vs XGBoost vs LLM — AUC/KS/Accuracy 표 + 막대 그래프

## 5. 배포 (Google Cloud)

### 컨테이너화
- `Dockerfile`: Python 3.11-slim + FastAPI + 모델 아티팩트 포함
- 추론 모델: XGBoost (즉시 배포)
- LLM은 Vertex AI Endpoint 또는 Cloud Run with GPU (옵션)

### Cloud Run 배포
```
deploy/cloudbuild.yaml — 컨테이너 빌드/푸시
deploy/service.yaml    — Cloud Run 서비스 정의
```

### 학습 (Vertex AI)
- `deploy/vertex_train.yaml`: A100/L4 인스턴스에서 QLoRA 학습
- 학습 데이터: GCS에 업로드된 train.parquet

## 6. 모델 비교 페이지
| 모델 | AUC | KS | Accuracy | F1 | 비교 |
|---|---|---|---|---|---|
| 로지스틱 회귀 (전통) | 0.70 | 0.30 | 0.81 | 0.55 | 기준 |
| XGBoost | 0.73 | 0.35 | 0.82 | 0.58 | +4.3% AUC |
| LLM (Qwen2.5-7B) | TBD | TBD | TBD | TBD | 목표 +7% |

(실제 수치는 학습 후 자동 갱신)

## 7. 작업 단계 (Execution Phases)

### ✅ Phase A: 세션 내 완성 (Now)
1. 프로젝트 스캐폴딩 + 의존성
2. 데이터 전처리 파이프라인 (`src/data/`)
3. 베이스라인 + XGBoost 학습 스크립트 (`src/models/`)
4. **실제 학습 수행** → metrics.json 산출
5. FastAPI 웹앱 + 한국어 프론트엔드
6. 비교 페이지 (실제 베이스라인 + XGBoost 수치)
7. Dockerfile + Cloud Run 설정
8. README + 한국어 문서
9. 로컬 git 초기화 + 첫 커밋

### ⏳ Phase B: 사용자 실행 (Out-of-Session)
1. **GitHub 푸시**: `gh auth login` → `gh repo create hwkim0527/CSS_rating2 --private` → `git push`
2. **GCP LLM 학습**: 제공된 `train_llm.py` + `vertex_train.yaml`을 Vertex AI에서 실행
   - 예상 비용: A100 1대 × 6시간 ≈ $20~30
3. **Cloud Run 배포**: `gcloud builds submit` → `gcloud run deploy`

## 8. 정직한 한계 명시
- 한 세션 내에서 7B 모델을 GPU 학습시켜 +10% 성능에 도달시키는 것은 불가능 (시간/리소스)
- GCP 학습/배포 및 GitHub 푸시는 사용자 자격증명이 필요하므로 명령어 키트로 제공
- LLM이 tabular data에서 GBM류를 능가하는 것은 일반적이지 않음 → 정직하게 비교 결과를 보여줌

## 9. 성공 기준
- [ ] 데이터 전처리: leakage 없이 train/val/test 산출
- [ ] XGBoost AUC ≥ 0.72 on test set
- [ ] FastAPI 서버 정상 기동, `/api/score` 응답 ≤ 200ms (XGBoost)
- [ ] 한국어 UI에서 입력 → 점수 산출 동작
- [ ] 비교 페이지에 실제 베이스라인/XGBoost 수치 표시
- [ ] Dockerfile로 컨테이너 빌드 성공
- [ ] git log에 의미있는 커밋 메시지
- [ ] LLM 학습 스크립트 dry-run 검증
