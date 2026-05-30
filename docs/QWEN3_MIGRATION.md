# Qwen2.5-7B → Qwen3-14B 마이그레이션 가이드

신용평가 sLLM 을 **Qwen2.5-7B → Qwen3-14B (QLoRA)** 로 교체했습니다.
학습은 사용자가 직접 **Colab** 에서 실행하고, 결과를 **Google Drive
(`내 드라이브/Colab Notebooks/Qwen3_fintech`)** 에 저장하며, 배포되는 신용평가
시스템은 그 Drive 의 학습된 어댑터를 내려받아 추론합니다.

## 왜 Qwen3-14B 인가

- 같은 크기 대비 약 1.5세대 성능 향상 — **Qwen3-14B ≈ Qwen2.5-32B 급**.
- STEM·수치 추론 개선 → 재무지표 기반 신용 판단에 직접 유리.
- Apache 2.0 (dense 모델) → 상업적 온프레미스 사용 자유.
- ⚠️ Qwen3 아키텍처는 **transformers>=4.51.0** 필요.

## 변경된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/training/llm_finetune.py` | 기본 모델 `Qwen/Qwen3-14B`, **체크포인트 자동 재개**(`--resume_from_checkpoint auto`), `--save_steps`/`--save_total_limit` 인자 추가, T4 기본값(batch 1 / accum 8) |
| `src/training/llm_eval.py` | 기본 모델 Qwen3-14B, metrics 키 `llm_qwen3_14b` |
| `src/web/llm_scoring.py` | 추론 base `Qwen/Qwen3-14B`, 어댑터 디렉터리 `artifacts/qwen3_lora`, **Drive 자동 다운로드** 연동, `CSS_LLM_ADAPTER_DIR` 로 경로 직접 지정 가능 |
| `src/web/download_model.py` (신규) | Google Drive 에서 어댑터 다운로드 (gdown / 서비스계정 두 방식) |
| `src/web/scoring.py` | LLM 모델명 라벨 `Qwen3-14B QLoRA` |
| `src/models/train.py`, `artifacts/metrics.json` | 비교표 placeholder 키/라벨 `llm_qwen3_14b` |
| `configs/training.yaml` | Qwen3-14B 하이퍼파라미터 + Drive 출력 경로 |
| `requirements.txt`, `src/training/requirements_llm.txt` | `gdown` 추가, transformers>=4.51 명시 |
| `notebooks/train_llm_colab_qwen3_14b.ipynb` (신규) | Colab 학습 노트북 (Drive 마운트→학습→체크포인트→평가→추론 테스트) |
| `Dockerfile.gpu` (신규) | Cloud Run GPU(L4)용 CUDA 이미지 (LLM 추론) |
| `deploy/cloudbuild_gpu.yaml` (신규) | Cloud Run GPU 배포 — LLM 활성화 + 폴더 ID/ADC 주입 |
| `deploy/setup_service_account.sh` (신규) | 런타임 SA 생성 + Drive 공유 안내 (키 파일 없음) |
| `deploy/entrypoint.sh` (신규) | 기동 시 Drive 어댑터 사전 다운로드 후 서버 기동 |
| `Dockerfile`, `render.yaml` | entrypoint 적용 + LLM 환경변수(기본 OFF) |
| `deploy/vertex_train.yaml`, `deploy/vertex_bootstrap.sh` | GCP 대안 경로도 Qwen3-14B 로 동기화 |

> 레거시 노트북 `train_llm_colab.ipynb`, `train_llm_colab_enterprise.ipynb` 는
> Qwen2.5 기준 그대로 보존(참고용). 신규 학습은 `train_llm_colab_qwen3_14b.ipynb` 사용.

## 1단계 — Colab 에서 학습 (사용자 직접 실행)

1. `notebooks/train_llm_colab_qwen3_14b.ipynb` 를 Colab 에서 엽니다.
2. **런타임 유형 → GPU**: L4(권장) 또는 A100. 무료 T4 도 가능(자동 보수 설정).
3. 셀을 위에서부터 실행:
   - Drive 마운트 → 저장 폴더 `Qwen3_fintech` 생성
   - 저장소 클론(GitHub 토큰 1회) → 의존성 설치(transformers>=4.51)
   - GPU 자동 감지로 batch/seq 설정
   - **학습**: 50스텝마다 체크포인트가 Drive 에 저장됨.
     런타임이 끊기면 학습 셀을 다시 실행 → **마지막 체크포인트에서 자동 재개**.
   - 평가(AUC/KS) → 추론 테스트
4. 학습 완료 시 최종 LoRA 어댑터가
   `내 드라이브/Colab Notebooks/Qwen3_fintech` 에 저장됩니다.

## 2단계 — 배포 시스템이 Drive 모델 사용

> **이 프로젝트의 Qwen3_fintech 폴더 ID: `1q6P-9a_U3bvln9XKeTKpe-3D6bNj-yYY`**
> (`render.yaml` 의 `CSS_LLM_DRIVE_FOLDER_ID` 에 이미 기입됨)
>
> ⚠️ **현재 이 폴더는 비공개(401)** 입니다. 배포 서버가 받으려면 아래 공유
> 설정이 필요합니다. 금융 모델이므로 **서비스 계정(방식 B)** 을 권장합니다.
>
> ⚠️ **CPU 호스트 주의**: Render free / HF Spaces CPU / Cloud Run 기본은 모두
> CPU 라 Qwen3-14B(4bit)를 **실행할 수 없습니다**. 그래서 `CSS_ENABLE_LLM=0`
> 을 기본으로 두어 XGBoost 로 정상 동작하게 했습니다. LLM 채점을 쓰려면
> **CUDA GPU 호스트**(예: HF Spaces T4, Cloud Run GPU, GCE GPU)에서
> `CSS_ENABLE_LLM=1` 로 켜세요.

배포 서버(신용평가 시스템)에서 환경변수로 어댑터를 가져옵니다.
컨테이너는 `deploy/entrypoint.sh` 가 기동 시 `CSS_ENABLE_LLM=1` + 폴더 ID 가
있으면 어댑터를 먼저 내려받고 서버를 띄웁니다(실패해도 XGBoost 로 기동).

### 방식 A — gdown (간편, 폴더를 링크 공유)
Drive 에서 `Qwen3_fintech` 폴더 → 공유 → "링크가 있는 모든 사용자".
노트북 마지막 셀이 출력하는 **폴더 ID** 를 사용:

```bash
export CSS_ENABLE_LLM=1
export CSS_LLM_BASE=Qwen/Qwen3-14B
export CSS_LLM_DRIVE_FOLDER_ID=<Qwen3_fintech 폴더 ID>
python -m src.web.download_model        # artifacts/qwen3_lora 로 받음
# 이후 웹앱 기동 — 첫 LLM 추론 시 자동으로도 받아짐
```

> ⚠️ 링크를 아는 누구나 모델을 받을 수 있어 **금융 모델에는 비권장**.

### 방식 B — 서비스 계정 (비공개, **이 프로젝트가 채택한 방식**)

배포 타깃: **GCP Cloud Run GPU(NVIDIA L4)**. 인증은 **키 파일 없이 런타임
서비스 계정 + ADC**(Application Default Credentials)로 한다 → JSON 키 유출 0.

1회 설정 (스크립트 제공):
```bash
PROJECT_ID=<your-proj> bash deploy/setup_service_account.sh
# → Drive API 활성화 + 런타임 SA(css-rating2-run@...) 생성 + SA 이메일 출력
```
그 다음 **Drive 웹에서 `Qwen3_fintech` 폴더를 출력된 SA 이메일에 '뷰어'로 공유**.

배포 (서울 리전은 GPU 미지원 → us-central1):
```bash
gcloud builds submit --config deploy/cloudbuild_gpu.yaml \
  --substitutions _REGION=us-central1,_RUNTIME_SA=css-rating2-run@<proj>.iam.gserviceaccount.com
```
`cloudbuild_gpu.yaml` 이 `Dockerfile.gpu`(CUDA 베이스)로 빌드하고, Cloud Run
서비스에 `CSS_ENABLE_LLM=1`, `CSS_LLM_GDRIVE_USE_ADC=1`, 폴더 ID 를 주입한다.
컨테이너 기동 시 `entrypoint.sh` 가 ADC 로 어댑터를 받아 Qwen3-14B 로 채점.

> 키 파일 방식이 필요하면(예: 비-GCP 호스트): SA JSON 키를 발급해
> `CSS_LLM_GDRIVE_SA_JSON=/secrets/sa.json` 로 지정. ADC 와 동일 코드 경로.

### 방식 C — 수동 배치
어댑터 폴더를 직접 `artifacts/qwen3_lora/` 에 복사 후 `CSS_ENABLE_LLM=1`.

## 환경변수 요약

| 변수 | 의미 | 기본값 |
|------|------|--------|
| `CSS_ENABLE_LLM` | LLM 채점 경로 활성화 | `0` |
| `CSS_LLM_BASE` | 베이스 모델 | `Qwen/Qwen3-14B` |
| `CSS_LLM_ADAPTER_DIR` | 어댑터 로컬 경로 | `artifacts/qwen3_lora` |
| `CSS_LLM_DRIVE_FOLDER_ID` | Drive 의 Qwen3_fintech 폴더 ID (다운로드 트리거) | (없음) |
| `CSS_LLM_GDRIVE_SA_JSON` | 서비스 계정 키 경로 (설정 시 비공개 다운로드) | (없음) |
| `CSS_LLM_GDRIVE_USE_ADC` | `1`이면 키 파일 없이 런타임 SA(ADC)로 Drive 접근 | `0` |

## 메모리 / GPU 주의

- Qwen3-14B 4bit QLoRA 추론 VRAM ≈ 10~12GB → 16GB GPU 1장이면 가능.
- 학습은 T4(16GB)에서 빠듯하므로 노트북이 batch=1/seq=384 로 자동 축소.
  여유 있는 결과를 원하면 L4/A100 사용.
- 추론도 GPU 필요(4bit). CPU 전용 서버라면 별도 양자화/서빙 검토 필요.

## 알려진 리스크 / 트레이드오프

- **Drive 에 체크포인트 직접 쓰기**: Colab 의 Drive(FUSE)는 잦은 소파일 쓰기에
  취약해 간헐적 I/O 지연이 날 수 있음. 단, 여기 체크포인트는 **LoRA 어댑터 +
  옵티마이저 상태(LoRA 파라미터 한정)** 라 수십 MB 수준 — 풀 모델이 아니므로
  부담이 작고, 무엇보다 **런타임 끊김 후 재개**를 가능케 하는 핵심이라 Drive
  저장을 기본으로 둔다. 만약 학습 중 Drive I/O 가 자주 멈추면, `--output_dir`
  를 `/content/qwen3_ckpts` 같은 **로컬 경로**로 바꿔 학습한 뒤 끝나고 Drive 로
  복사하면 됨 (대신 그 경우 끊기면 처음부터 재학습).
- **단일 토큰 분류**: "부실"/"정상" 의 첫 토큰이 같으면 logit 비교가 무의미.
  `llm_eval.py`/`llm_scoring.py` 에 `pos_id != neg_id` assert 를 넣어 silent
  failure 를 막음 (Qwen3 는 Qwen2 토크나이저 계열이라 정상 분해될 가능성 높음).

## 검증된 것 / 검증 안 된 것

- ✅ 코드/노트북/설정의 모델·경로·키 일관성 (정적 점검 통과).
- ⚠️ 실제 Qwen3-14B 학습·추론은 **GPU 환경에서 미실행** — Colab 에서 1회
  엔드투엔드 실행으로 검증 필요(특히 토크나이저의 "부실"/"정상" 단일 토큰
  분해, Qwen3 `<think>` 억제 동작).
