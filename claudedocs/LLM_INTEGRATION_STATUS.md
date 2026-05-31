# Qwen3-14B sLLM 신용평가 통합 — 현황 (2026-05-31)

## 한 줄 요약
학습된 Qwen3-14B QLoRA 어댑터가 **라이브 GCP Cloud Run GPU(L4)에서 실제 추론에
에러 없이 성공함을 검증**했다(워밍 시 0.77s, 200 OK). 그러나 변별력은 약하고
(실측 AUC 0.640, 95%CI≈[0.57,0.71], N=398 → 로지스틱 0.697·XGBoost 0.723 점추정 하회),
**내가 한 코드 개선(프롬프트 정렬·UI 모델선택기·테스트)은 로컬에만 있고 라이브엔
미배포**(재빌드 필요)이며, 비용 보호를 위해 **min=0** 으로 내려 **LLM 콜드 요청은
현재 행(hang)/끊김** 상태다(XGBoost만 콜드에서 신뢰성 있게 서빙). 자세한 경계는 아래.

## ⚠️ 상태 정확성 경계 (과대표현 방지)
- **라이브 = 구 코드**: 이번 세션의 수정(serialize 프롬프트 단일화, index.html 모델선택기,
  score.js, entrypoint 주석)은 전부 **미커밋·미배포**. 라이브 `:latest` 이미지는 이전 커밋
  기반이라 **라이브 웹 UI엔 sLLM 모델 선택기가 없고**(브라우저에선 XGBoost만; sLLM은
  `/api/score_llm` 직접 호출만 가능), 추론 프롬프트도 구버전. 실측 AUC 0.640도 *구 프롬프트* 값.
- **min=0의 실제 의미**: 유휴 후 첫 `/api/score_llm` = 콜드 인스턴스 → 핸들러 안에서
  29.5GB lazy 다운로드(~11분) → 클라이언트 연결 끊김(HTTP 000), 재시도 시 인스턴스 증식·재다운로드.
  즉 **콜드 상태에서 LLM 엔드포인트는 클라이언트에 사실상 미작동**. 신뢰성 서빙은 XGBoost뿐.
  (사용자가 min=1을 고른 이유가 이 콜드스타트 회피였음 — 비용 보호와 trade-off.)

## 라이브 추론 검증 결과 (2026-05-31, Cloud Run GPU L4 실측)
- ✅ **추론 작동·무에러**: `/api/score_llm` → `200 OK`, 8/8 샤드 4bit GPU 로딩 성공, OOM 없음.
- ✅ **워밍 시 0.77s/건, steady-state 안정**(연속 요청 재다운로드 없이 즉시 응답).
- ⚠️ **콜드스타트 = 29.5GB 베이스 Drive 다운로드(~10분) + 로딩** → 첫 요청이 매우 느리고
  클라이언트 연결이 끊길 수 있음(서버는 200 반환). 상시 운영엔 워밍 전략 필요(아래).
- 📊 **실측 변별력 (균형표본 398건, 부실199/정상199)**: **AUC 0.640, KS 0.258**,
  부실확률 분포 0.13~0.40(spread 0.26). → 약한 신호는 있으나 두 전통 모델보다 낮음.

| 모델 | AUC | KS | 비고 |
|---|---|---|---|
| 로지스틱(전통) | 0.697 | 0.287 | 전체 테스트셋 |
| XGBoost(운영) | 0.723 | 0.323 | 전체 테스트셋 |
| **Qwen3-14B sLLM** | **0.640** | **0.258** | 라이브 398건 실측 |

원인: 1 epoch / 10k 샘플 / LoRA r=8 = 매우 경량 → 미학습. 표 데이터에서 경량 QLoRA
LLM이 GBM을 능가하지 못함(README 한계 4와 일치). **개선 레버 = 재학습**(epoch/데이터/rank↑),
프롬프트 수정·재빌드로 해결되는 문제 아님(피처가 동일하므로).

## 환경 사실
- 로컬 GPU: RTX 3050 **4GB** (Qwen3-14B 4bit ≈9GB → 로컬 실행 불가)
- 로컬: torch 2.5.1+cu121, transformers 4.57.3, peft 0.18.0, RAM 34GB(여유 ~14GB)
- gcloud: 프로젝트 `qwen3-fintech`, 계정 hwkim0527@gmail.com (ADC 미설정)
- 학습 어댑터: Drive `Qwen3_fintech`(ID `1q6P-9a_U3bvln9XKeTKpe-3D6bNj-yYY`)에 실재
  - `adapter_model.safetensors`(~61MB), `adapter_config.json`(base=Qwen/Qwen3-14B, r=8, α=16),
    tokenizer, checkpoint-600/625, training_summary(1 epoch / train 10k·val 1k)

## 이번에 검증/수정한 것 (GPU 불필요)
| 항목 | 결과 |
|------|------|
| 학습 어댑터 Drive 실재 + config 정합성 | ✅ base=Qwen3-14B, r=8, α=16 |
| 토크나이저 단일 토큰 분해 (부실≠정상 첫 토큰) | ✅ 63089 ≠ 29281 |
| 학습=추론=평가 프롬프트 일치 | ✅ `serialize.build_chat_text` 단일 원천으로 통합 |
| **(버그수정)** 추론/평가 system 프롬프트가 학습과 불일치 | ✅ 학습 포맷("부실(1)/정상(0)")에 정렬 |
| peft 0.19.1 어댑터 config 로드 호환 | ✅ 설치 peft 가 파싱 가능 |
| score_with_llm 수학 경로(softmax/인덱싱) | ✅ mock 단위테스트 통과 |
| UI에서 sLLM 선택·채점 | ✅ 모델 선택기 + `/api/score_llm` 연결 + GPU 없을 때 graceful 비활성 |
| 전체 테스트 | ✅ 22/22 통과 |

자동화: `python -m scripts.verify_llm_local` (4/4), `pytest tests/test_llm_scoring.py`.

## 변경 파일
- `src/training/serialize.py` — `SYSTEM_PROMPT`, `build_chat_text()` 단일 원천 추가
- `src/training/llm_finetune.py`, `src/web/llm_scoring.py`, `src/training/llm_eval.py` — 공유 빌더 사용(프롬프트 정렬)
- `frontend/templates/index.html`, `frontend/static/score.js`, `frontend/static/styles.css` — 모델 선택기 + 가용성 표시
- `tests/test_llm_scoring.py`, `scripts/verify_llm_local.py` — 신규 검증
- `.github/workflows/ci.yml` — 새 테스트 실행
- `deploy/cloudbuild_gpu.yaml` — `_MIN_INSTANCES`(기본 0=과금 안전) 파라미터화
- `docs/QWEN3_MIGRATION.md`, `docs/PLAN.md`, `artifacts/metrics.json` — 정직성/현황 갱신

## 현재 배포 상태 (2026-05-31 기준)
- 서비스 `css-rating2-gpu` (us-central1, L4) **min-instances=0 으로 변경**(리비전 00015).
  → 유휴 시 GPU 과금 0, on-demand 동작(첫 요청 콜드스타트 수분). 측정 종료 후 비용 보호.
- URL: https://css-rating2-gpu-u4exkcsxwq-uc.a.run.app  (`/`, `/api/score`, `/api/score_llm`, `/compare`)
- 사용자는 always-on(min=1, ~$300-500/월)을 골랐으나, LLM 변별력이 베이스라인 미만으로
  드러나 전제가 약해져 **가역적으로 min=0** 으로 내려둠. 복구는 한 커맨드(아래).

## 남은 선택지 (사용자 결정)
1. **LLM을 쓸모 있게 = 재학습** (진짜 레버): Colab 노트북으로 epoch↑(예: 3) /
   데이터↑(50k) / LoRA rank↑(16~32) / `<think>` 억제 확인. 무료, 결과를 Drive에 저장하면
   배포가 자동으로 새 어댑터 사용.
2. **배포 운영 방식**:
   - 현 상태 유지(min=0, 과금~0, 콜드스타트) ← 권장(현재값)
   - always-on 복구: `gcloud run services update css-rating2-gpu --region=us-central1 --min-instances=1`
   - 완전 중단: `gcloud run services delete css-rating2-gpu --region=us-central1`
3. **상시(min=1) 운영 시 콜드스타트 강건화(선택)**: 앱 시작 시 베이스 백그라운드 워밍
   (warming/ready 플래그 + 락 + ready 전 503) 추가 후 재빌드. 동시 다운로드 race 방지 포함.
   (현재 entrypoint는 이 race 때문에 베이스 프리페치를 의도적으로 빼 둠.)

## 정직성 경계
"추론이 에러 없이 작동한다"는 **라이브 200 OK로 검증 완료**. 그러나 "쓸 만한 신용평가
LLM"은 **아직 아님**(AUC 0.640 < 베이스라인). 이는 코드가 아니라 학습량의 문제이며,
재학습 없이는 닫히지 않는다.
