# GitHub 비공개 저장소 푸시 — 실행 명령

`hwkim0527/CSS_rating2` 저장소를 비공개로 생성하고 푸시하는 정확한 순서입니다.

## 1. GitHub CLI 인증 확인

```bash
gh auth status
```

- 결과에 `Logged in to github.com as hwkim0527` 가 보이면 다음 단계로 진행.
- 그렇지 않으면 한 번만 인증:
  ```bash
  gh auth login
  # → GitHub.com, HTTPS, Login with a web browser
  ```

## 2. 로컬 저장소 상태 확인

이 디렉토리에서 실행:
```bash
cd /f/project/신용평가_클로드/CSS_rating2
git status
git log --oneline | head -10
```

## 3. 비공개 저장소 생성 + 푸시 (한 줄)

```bash
gh repo create hwkim0527/CSS_rating2 \
  --private \
  --source=. \
  --remote=origin \
  --description "AI 신용평가시스템 — XGBoost + Qwen3-14B QLoRA + FastAPI" \
  --push
```

## 4. 푸시 확인

```bash
gh repo view hwkim0527/CSS_rating2 --web
```

## 만약 저장소가 이미 존재한다면

```bash
git remote add origin https://github.com/hwkim0527/CSS_rating2.git
git branch -M main
git push -u origin main
```

## SSH 키 사용 시
```bash
git remote set-url origin git@github.com:hwkim0527/CSS_rating2.git
git push -u origin main
```
