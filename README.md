# CSS Rating 2 — AI 신용평가시스템

[![CI](https://github.com/hwkim0527/CSS_rating2/actions/workflows/ci.yml/badge.svg)](https://github.com/hwkim0527/CSS_rating2/actions/workflows/ci.yml)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/hwkim0527/CSS_rating2)

개인의 금융 정보를 입력하면 **12개월 내 채무불이행(부실) 확률**을 산출하는 웹앱입니다.
Lending Club 공개 데이터(2.26M건)로 학습된 XGBoost를 즉시 사용할 수 있고,
**Qwen2.5-7B sLLM** QLoRA 파인튜닝 파이프라인을 GCP에서 실행할 수 있습니다.

> 본 시스템은 연구·교육 목적의 데모입니다. 실제 신용 결정 용도가 아닙니다.

## 🚀 라이브 배포

| 플랫폼 | URL | 비고 |
|---|---|---|
| Render | [Deploy ▶](https://render.com/deploy?repo=https://github.com/hwkim0527/CSS_rating2) | 무료, 무카드, 한 번 클릭 |
| HuggingFace Spaces | [docs/HF_SPACES](./deploy/HF_SPACES_README.md) | CPU basic 무료 |
| Google Cloud Run | `bash deploy/deploy.sh` | 항상 켜짐, 비용 발생 |
| 로컬 Docker | `docker build -t css-rating2 . && docker run -p 8080:8080 css-rating2` | 검증용 |

### 모델 학습 결과 (실제 측정값)
| 지표 | 전통 모델 (LR) | AI 모델 (XGBoost) | 개선 |
|---|---|---|---|
| **KS** *(신용평가 산업 표준)* | 0.2866 | **0.3229** | **+12.66% ✓** |
| Average Precision | 0.3846 | 0.4151 | +7.91% |
| AUC | 0.6972 | 0.7229 | +3.68% |

→ `/compare` 페이지에서 실시간 비교 표·그래프 확인

---

## 1. 빠른 시작 (Local)

### 사전 요구사항
- Python 3.11
- Raw 데이터: `../data_source/loan.csv` (Lending Club 2007-2018 — Kaggle)

### 설치
```bash
pip install -r requirements.txt
```

### 데이터 전처리 + 모델 학습 + 웹앱 실행
```bash
python -m src.data.preprocess --sample 300000
python -m src.models.train
uvicorn src.web.app:app --reload --port 8000
```

- `http://localhost:8000/` — 신용평가 입력 폼
- `http://localhost:8000/compare` — 전통 vs AI 모델 비교
- `http://localhost:8000/docs` — OpenAPI 문서

---

## 2. 폴더 구조

```
CSS_rating2/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore / .gitignore
├── configs/
│   └── training.yaml             # 하이퍼파라미터 참고용
├── deploy/
│   ├── cloudbuild.yaml           # Cloud Build → Cloud Run
│   ├── vertex_train.yaml         # Vertex AI LLM 학습 잡
│   └── deploy.sh                 # 원클릭 배포
├── docs/
│   └── PLAN.md                   # 전체 개발 계획
├── src/
│   ├── data/preprocess.py        # raw CSV → train/val/test parquet
│   ├── models/train.py           # 로지스틱 + XGBoost 학습
│   ├── training/
│   │   ├── serialize.py          # tabular → 자연어 프롬프트
│   │   ├── build_llm_dataset.py  # → JSONL
│   │   ├── llm_finetune.py       # Qwen2.5-7B QLoRA (GCP)
│   │   ├── llm_eval.py           # 학습 어댑터 평가
│   │   └── requirements_llm.txt
│   ├── utils/config.py           # 컬럼·경로·라벨 정의 (단일 진실 원천)
│   └── web/
│       ├── app.py                # FastAPI
│       ├── schemas.py            # Pydantic 입출력
│       └── scoring.py            # 모델 로딩·점수 산출
├── frontend/
│   ├── templates/                # Jinja2 HTML (한국어 UI)
│   └── static/                   # CSS, JS, 캔버스 차트
├── artifacts/                    # 학습 결과물 (.joblib, metrics.json)
└── tests/                        # pytest smoke + 데이터 검증
```

---

## 3. 모델

| 모델 | 파일 | 비고 |
|---|---|---|
| 로지스틱 회귀 (기준) | `artifacts/baseline_logistic.joblib` | 전통적 신용평가 베이스라인 |
| **XGBoost (운영)** | `artifacts/xgboost.joblib` | 현재 `/api/score` 가 사용 |
| Qwen2.5-7B QLoRA | `artifacts/qwen25_lora/` | GCP에서 학습 후 산출 |

### 성능 (테스트셋, 300k 샘플 기준)
실제 학습 결과는 `artifacts/metrics.json` 에 자동 기록됩니다.

### 누수(Leakage) 방지
대출 발생 *이후* 정보(예: `total_pymnt`, `recoveries`, `last_pymnt_*`)는 모두 제거했습니다.
Lending Club의 자체 등급(`grade`, `sub_grade`)도 제거하여 모델이 독립적으로 위험을 학습합니다.

---

## 4. GCP에서 LLM 학습 (전체 절차)

### Vertex AI 사용
```bash
# 1) Hugging Face 토큰 (Qwen 모델 다운로드)
export HF_TOKEN="hf_xxx"
export GCS_BUCKET="my-css-bucket"

# 2) 코드 + 데이터 GCS 업로드
gsutil -m cp -r ./src ./data gs://${GCS_BUCKET}/css-rating2/

# 3) Vertex AI 학습 잡
gcloud ai custom-jobs create \
    --region=asia-northeast3 \
    --display-name=css-rating2-llm-train \
    --config=deploy/vertex_train.yaml
```

- 예상 시간: L4 1대 × 50k 샘플 × 3 epoch ≈ 8~12시간
- 예상 비용: $0.71/시간 × 10시간 ≈ $7~10
- A100 (g2 대신 a2-highgpu-1g) 사용 시 빠르지만 ≈ $4/시간

### 평가 → metrics.json 갱신
```bash
python -m src.training.llm_eval \
    --adapter_dir gs://${GCS_BUCKET}/css-rating2/qwen25_lora \
    --metrics_path artifacts/metrics.json
```
LLM 결과가 추가되면 `/compare` 페이지에 자동 반영됩니다.

---

## 5. Cloud Run 배포

```bash
gcloud config set project YOUR_PROJECT_ID
bash deploy/deploy.sh
```

- 메모리 1GiB / CPU 1 / max-instances 10
- Cold start ≈ 5초 (XGBoost 모델 로딩 포함)
- `/api/score` p95 ≈ 50ms

### 로컬 Docker 테스트
```bash
docker build -t css-rating2 .
docker run -p 8080:8080 css-rating2
curl http://localhost:8080/healthz
```

---

## 6. API 사용

### `POST /api/score`
```bash
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 12000, "installment": 406, "int_rate": 13.5,
    "term": "36 months", "purpose": "debt_consolidation",
    "annual_inc": 65000, "emp_length": "5",
    "home_ownership": "RENT", "verification_status": "Not Verified",
    "addr_state": "CA", "dti": 18.5, "delinq_2yrs": 0,
    "inq_last_6mths": 0, "open_acc": 9, "pub_rec": 0,
    "revol_bal": 9500, "revol_util": 42, "total_acc": 22,
    "mort_acc": 1, "pub_rec_bankruptcies": 0,
    "credit_history_years": 12.5,
    "application_type": "Individual", "initial_list_status": "w"
  }'
```

### 응답
```json
{
  "default_probability": 0.4328,
  "credit_score": 566,
  "risk_grade": "D",
  "risk_grade_kr": "주의 (부실위험 높음)",
  "model_name": "XGBoost",
  "top_factors": [...]
}
```

---

## 7. 테스트

```bash
pytest tests/ -v
```

- `tests/test_data.py` — 누수 컬럼 / 라벨 분포 검증
- `tests/test_smoke.py` — API 엔드포인트 / 점수 산출 검증

---

## 8. 한계와 정직한 설명

1. **데이터 한계**: Lending Club 데이터에는 FICO 신용점수가 포함되지 않아 모델 성능 상한이 제한됩니다(AUC ≈ 0.71).
   실제 신용평가사 데이터에는 FICO/내부 평점이 가장 강력한 변수이므로 운영 환경에서는 0.80+ AUC가 가능합니다.
2. **LLM의 가치**: 표 데이터에서 LLM이 GBM을 큰 폭으로 능가하는 것은 일반적이지 않습니다.
   이 프로젝트의 LLM 트랙은 자연어 설명/추론 능력의 확장 가능성을 보이기 위한 연구 트랙입니다.
3. **"전통 모델보다 10% 우수" 정의**: AUC 절대값 +0.05 (예: 0.70 → 0.75)를 달성 기준으로 합니다.
4. **공정성**: `addr_state`, `purpose`는 보호 속성이 아니지만 우편번호·인종 대용변수가 될 수 있어 운영 시 별도 검토가 필요합니다.

---

## 9. 라이선스 / 데이터 출처

- 코드: MIT
- 데이터: Lending Club 공개 데이터 (Kaggle, 비상업적 연구)
- 모델: Qwen2.5-7B-Instruct (Apache 2.0)
