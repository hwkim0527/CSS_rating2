---
title: CSS Rating 2
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: AI 신용평가시스템 — XGBoost + Qwen3-14B QLoRA + FastAPI
---

# CSS Rating 2 — HuggingFace Spaces 배포

이 파일은 HuggingFace Spaces에 배포할 때 사용하는 README입니다.

## 배포 방법

1. https://huggingface.co/new-space 접속
2. **Owner**: hwkim0527 (또는 본인 계정)
3. **Space name**: `css-rating2`
4. **License**: MIT
5. **SDK**: Docker → "From a template" 선택 안함, "Dockerfile" 선택
6. **Hardware**: CPU basic (free)
7. **Visibility**: Private (선택)
8. 생성 후, 이 저장소의 모든 파일을 Space에 push
9. `README.md`를 본 파일(`deploy/HF_SPACES_README.md`)로 교체

또는 한 줄:
```bash
git remote add hf https://huggingface.co/spaces/hwkim0527/css-rating2
cp deploy/HF_SPACES_README.md README.md
git add README.md && git commit -m "HF Spaces config"
git push hf main
```

5분 내에 https://hwkim0527-css-rating2.hf.space 에서 접근 가능합니다.
