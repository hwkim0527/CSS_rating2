# Qwen3-14B sLLM 신용평가 통합 — 현황 (2026-05-31)

## ★ 최종 확정 (2026-06-02) — Qwen3-14B(50k/r=16)을 운영 최종 모델로 확정
사용자 결정으로 현재까지 학습된 Qwen3 어댑터를 **최종 모델로 확정**, 개발 마무리.
신용평가 시스템(라이브)이 이 모델로 채점하며, 문서·설정·라벨을 "최종" 으로 정리하고
GitHub 에 반영. 성능은 아래(로지스틱 능가 / XGBoost 대등) 그대로. 이하는 경위 기록.

## ★ 경위 (2026-06-01) — 재학습 모델 적용
사용자가 v2 재학습(50k / LoRA r=16,α32). Colab 끊김으로 최종 저장은 미완이나
**checkpoint-2500**(완전한 어댑터)을 배포에 반영. 런타임 SA 가 v2 폴더에 미공유라,
SA 접근 가능한 `1q6P` 하위에 `v2_step2500_adapter` 폴더를 만들고 어댑터 2파일을
Drive 내부 복사(상속 접근)해 해결. **라이브 실측 AUC 0.763 / KS 0.420 (100건)** — v1(0.640)
대비 대폭 개선, 로지스틱(0.697) 능가. **XGBoost 와는 동일 100건 기준 대등**(LLM 0.763 vs
XGBoost 0.759 — 이전 "능가"는 LLM-100건 vs XGB-전체199k 비교 오류였고, 공정 비교는 동률).
프론트엔드도 전면 재디자인("정밀 신용 계기판") 후 라이브 반영.
아래 v1 기준의 옛 서술은 이력 보존용.

## 한 줄 요약
학습된 Qwen3-14B QLoRA 어댑터가 **라이브 GCP Cloud Run GPU(L4)에서 실제 추론에
에러 없이 성공함을 검증**했다(워밍 시 0.77s, 200 OK). 그러나 변별력은 약하고
(실측 AUC 0.640, 95%CI≈[0.57,0.71], N=398 → 로지스틱 0.697·XGBoost 0.723 점추정 하회),
**내가 한 코드 개선(프롬프트 정렬·UI 모델선택기·테스트)은 로컬에만 있고 라이브엔
미배포**(재빌드 필요)이며, 비용 보호를 위해 **min=0** 으로 내려 **LLM 콜드 요청은
현재 행(hang)/끊김** 상태다(XGBoost만 콜드에서 신뢰성 있게 서빙). 자세한 경계는 아래.

## 재빌드·재배포 완료 (rev 00016, 2026-05-31) — 경계 해소됨
이전 경계(라이브=구코드, min=0 콜드 미작동)는 **재빌드+재배포로 모두 해소**됨:
- **라이브 = 신 코드**: 3개 커밋(프롬프트 단일화·UI 모델선택기·워밍 조율·리뷰수정 15건)을
  빌드해 배포(rev `css-rating2-gpu-00016-ztw`). 라이브 웹 UI에 **모델 선택기(XGBoost/Qwen3-14B sLLM) 노출 확인.**
- **워밍 조율 검증(happy-path 라이브)**: 기동 후 warm_status warming(~8.5분, 베이스 29.5GB
  Drive 다운로드+로딩) → **ready**. ready 후 `/api/score_llm` 저위험 200(3.5s) / 고위험 200(0.78s)
  / 3차 0.74s — **HTTP 000 연결 끊김 없음.**
  - ⚠️ 정직성: "ready 전 요청 즉시 503" 동작은 **단위테스트(mock)로 검증**했고 라이브로는
    warming→ready happy-path만 확인함(warming 창에 라이브 요청을 직접 때리지는 않음).
- **min=1 상시(사용자 선택)** + concurrency=1(라이브 확인, 단일 GPU forward 직렬화 OOM 방지)
  + 베이스 모델 Drive 폴더 기본값(HF 429 우회) + 다운로드 완료 센티넬.
  - ⚠️ 센티넬의 **부분 다운로드 자가복구(중단→resume)**는 코드/단위 로직 검증이며, 실제
    중단 시나리오를 라이브로 재현하진 않음(happy-path 다운로드+센티넬 기록은 라이브 OK).
- ⚠️ **여전한 정직성 경계**: 배포된 모델은 **기존 v1 어댑터(r=8/10k)** 라 변별력은 그대로 약함
  (출력 0.173/0.178, AUC≈0.640). 품질 개선은 **v2 재학습 노트북 실행**으로만 달성(코드/서빙 문제 아님).
  그리고 **min=1 상시 GPU = ~$300-500/월 과금** 지속 중(아래 teardown 참조).

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

## 현재 배포 상태 (2026-05-31 기준, rev 00016)
- 서비스 `css-rating2-gpu` (us-central1, L4) **min-instances=1 (상시·사용자 선택)**, concurrency=1.
- URL: https://css-rating2-gpu-u4exkcsxwq-uc.a.run.app  (`/`, `/api/score`, `/api/score_llm`, `/api/llm_status`, `/compare`)
- LLM 워밍 검증 완료(ready, 콜드 요청 끊김 없음). **min=1 상시 GPU = ~$300-500/월 과금 진행 중.**
- 비용 조절(한 커맨드):
  - scale-to-zero(유휴 과금~0): `gcloud run services update css-rating2-gpu --region=us-central1 --min-instances=0`
  - 완전 중단: `gcloud run services delete css-rating2-gpu --region=us-central1`

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
