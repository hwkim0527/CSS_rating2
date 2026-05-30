"""학습된 Qwen3-14B LoRA 어댑터를 Google Drive 에서 내려받는다.

배포되는 신용평가 시스템이 hwkim0527 의 Google Drive
"Colab Notebooks/Qwen3_fintech" 폴더에 저장된 어댑터를 사용할 수 있게 한다.

두 가지 방식을 환경변수로 전환:

  1) gdown (간편) — 폴더를 "링크가 있는 모든 사용자" 로 공유한 뒤:
         export CSS_LLM_DRIVE_FOLDER_ID=<폴더 ID>
     ⚠️ 링크를 아는 누구나 모델을 받을 수 있으므로, 금융 모델에는 권장하지 않음.

  2) 서비스 계정 (비공개·권장) — 폴더를 서비스 계정 이메일과 공유한 뒤:
         export CSS_LLM_DRIVE_FOLDER_ID=<폴더 ID>
         export CSS_LLM_GDRIVE_SA_JSON=/secrets/sa.json
     (추가 설치: pip install google-api-python-client google-auth)

CLI (배포 시작 시 미리 받기):
    python -m src.web.download_model            # artifacts/qwen3_lora 로 받음
    python -m src.web.download_model --target /models/qwen3_lora

폴더 ID 는 Drive 폴더 URL 의 .../folders/<여기> 부분이다.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("download_model")

# 어댑터로 인정하기 위해 반드시 있어야 하는 파일
REQUIRED_FILE = "adapter_config.json"


def _flatten_into(target: Path) -> None:
    """gdown 이 하위 폴더로 받았을 경우, adapter_config.json 이 있는 폴더의
    내용을 target 바로 아래로 끌어올린다."""
    if (target / REQUIRED_FILE).exists():
        return
    matches = list(target.rglob(REQUIRED_FILE))
    if not matches:
        return
    src_dir = matches[0].parent
    if src_dir == target:
        return
    log.info("어댑터 파일을 %s → %s 로 평탄화합니다", src_dir, target)
    for item in src_dir.iterdir():
        dest = target / item.name
        if dest.exists():
            continue
        shutil.move(str(item), str(dest))


def _download_with_gdown(folder_id: str, target: Path) -> None:
    import gdown  # requirements.txt 에 포함
    from gdown.exceptions import DownloadError

    target.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    log.info("gdown 으로 폴더 다운로드: %s", url)
    try:
        gdown.download_folder(url=url, output=str(target), quiet=False, use_cookies=False)
    except DownloadError as e:
        raise RuntimeError(
            f"Drive 폴더에 접근할 수 없습니다 (folder_id={folder_id}).\n"
            "  원인: 폴더가 비공개이거나 링크 공유가 꺼져 있습니다.\n"
            "  해결 (둘 중 하나):\n"
            "   (A) gdown 방식 — Drive 에서 Qwen3_fintech 폴더 우클릭 → 공유 →\n"
            "       '링크가 있는 모든 사용자(뷰어)' 로 변경.\n"
            "   (B) 서비스 계정(권장) — 폴더를 서비스 계정 이메일과 공유하고\n"
            "       CSS_LLM_GDRIVE_SA_JSON 에 키 파일 경로를 설정.\n"
            f"  (원본 오류: {e})"
        ) from e
    _flatten_into(target)


def _download_with_service_account(folder_id: str, sa_json: str, target: Path) -> None:
    """sa_json 이 비어 있으면 Application Default Credentials(ADC)를 사용한다.

    Cloud Run/GCE 에서는 런타임 서비스 계정으로 ADC 가 자동 제공되므로, 키
    파일 없이 폴더를 그 서비스 계정 이메일과 공유하기만 하면 된다(키 유출 0).
    """
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "서비스 계정 방식에는 추가 패키지가 필요합니다: "
            "pip install google-api-python-client google-auth"
        ) from e
    import io

    if sa_json:
        from google.oauth2 import service_account

        log.info("서비스 계정 키 파일로 인증: %s", sa_json)
        creds = service_account.Credentials.from_service_account_file(sa_json, scopes=SCOPES)
    else:
        import google.auth

        log.info("Application Default Credentials(런타임 서비스 계정)로 인증")
        creds, _ = google.auth.default(scopes=SCOPES)
    service = build("drive", "v3", credentials=creds)

    target.mkdir(parents=True, exist_ok=True)
    query = f"'{folder_id}' in parents and trashed=false"
    resp = service.files().list(
        q=query, fields="files(id, name, mimeType)", pageSize=1000
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise RuntimeError(
            f"폴더가 비었거나 서비스 계정에 공유되지 않았습니다 (folder_id={folder_id})."
        )
    for f in files:
        if f["mimeType"] == "application/vnd.google-apps.folder":
            continue  # 어댑터는 평면 폴더라 하위 폴더는 건너뜀
        dest = target / f["name"]
        log.info("다운로드: %s", f["name"])
        request = service.files().get_media(fileId=f["id"])
        with io.FileIO(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()


def download_adapter_from_drive(target_dir: str | Path) -> Path:
    """환경변수 설정에 따라 어댑터를 target_dir 로 내려받는다.

    이미 받아져 있으면(adapter_config.json 존재) 다시 받지 않는다.
    """
    target = Path(target_dir)
    folder_id = os.environ.get("CSS_LLM_DRIVE_FOLDER_ID", "")
    sa_json = os.environ.get("CSS_LLM_GDRIVE_SA_JSON", "")
    use_adc = os.environ.get("CSS_LLM_GDRIVE_USE_ADC", "0") == "1"

    if (target / REQUIRED_FILE).exists():
        log.info("어댑터가 이미 존재합니다: %s (다운로드 생략)", target)
        return target
    if not folder_id:
        raise RuntimeError(
            "CSS_LLM_DRIVE_FOLDER_ID 가 설정되지 않았습니다. Drive 폴더 ID 를 지정하세요."
        )

    if sa_json or use_adc:
        # 비공개·권장 경로. sa_json 있으면 키 파일, 없으면 런타임 SA(ADC).
        log.info("서비스 계정 방식(비공개)으로 다운로드합니다.")
        _download_with_service_account(folder_id, sa_json, target)
    else:
        log.info("gdown(공개 링크) 방식으로 다운로드합니다.")
        _download_with_gdown(folder_id, target)

    if not (target / REQUIRED_FILE).exists():
        raise RuntimeError(
            f"다운로드는 끝났지만 {REQUIRED_FILE} 를 찾지 못했습니다. "
            f"폴더 ID/공유 설정을 확인하세요: {target}"
        )
    log.info("어댑터 다운로드 완료 → %s", target)
    return target


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.utils.config import ARTIFACTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(ARTIFACTS_DIR / "qwen3_lora"),
                        help="어댑터를 저장할 로컬 경로.")
    args = parser.parse_args()
    download_adapter_from_drive(args.target)


if __name__ == "__main__":
    main()
