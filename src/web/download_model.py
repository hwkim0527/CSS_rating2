"""학습된 Qwen3-14B LoRA 어댑터(및 선택적으로 베이스 모델)를 Google Drive 에서
내려받는다.

배포되는 신용평가 시스템이 hwkim0527 의 Google Drive 폴더에 저장된 모델을
사용할 수 있게 한다. GCP Cloud Run IP 대역은 HuggingFace Hub 에서 429(Too Many
Requests)를 자주 받으므로, 베이스 모델(Qwen3-14B, ~28GB)도 HF 가 아니라 Drive
에서 받는 경로를 지원한다(HF 완전 우회).

두 가지 인증 방식(환경변수로 전환):
  1) gdown — 폴더를 "링크가 있는 모든 사용자" 로 공유 (간편, 보안 약함)
  2) 서비스 계정 / ADC — 폴더를 SA 이메일과 공유 (비공개·권장)
     CSS_LLM_GDRIVE_SA_JSON(키파일) 또는 CSS_LLM_GDRIVE_USE_ADC=1(런타임 SA)

CLI:
    python -m src.web.download_model                         # 어댑터
    python -m src.web.download_model --target /m/qwen3_lora  # 경로 지정
    python -m src.web.download_model --base                  # 베이스 모델
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("download_model")

# 폴더 종류별 "이게 있으면 (적어도) 받기 시작했다" 로 보는 대표 파일
ADAPTER_REQUIRED_FILE = "adapter_config.json"
BASE_REQUIRED_FILE = "config.json"
# ⚠️ 완료 판정은 대표 파일이 아니라 이 센티넬로 한다. 대표 파일(config.json 등)은
#    작아서 다운로드 초반에 떨어지므로, 거대 샤드(safetensors)가 중간에 끊긴
#    부분 다운로드를 '완료'로 오인할 수 있다(영구 손상 캐시). 모든 파일을 받은
#    뒤에만 이 센티넬을 기록하고, 스킵 판정도 센티넬 존재로만 한다.
COMPLETE_SENTINEL = ".download_complete"


def _flatten_into(target: Path, required_file: str) -> None:
    """gdown 이 하위 폴더로 받았을 경우, required_file 이 있는 폴더의 내용을
    target 바로 아래로 끌어올린다."""
    if (target / required_file).exists():
        return
    matches = list(target.rglob(required_file))
    if not matches:
        return
    src_dir = matches[0].parent
    if src_dir == target:
        return
    log.info("파일을 %s → %s 로 평탄화합니다", src_dir, target)
    for item in src_dir.iterdir():
        dest = target / item.name
        if dest.exists():
            continue
        shutil.move(str(item), str(dest))


def _download_with_gdown(folder_id: str, target: Path, required_file: str) -> None:
    import gdown
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
            "  해결: (A) 폴더를 '링크가 있는 모든 사용자(뷰어)' 로 공유, 또는\n"
            "        (B) 서비스 계정 이메일과 공유 후 CSS_LLM_GDRIVE_USE_ADC=1.\n"
            f"  (원본 오류: {e})"
        ) from e
    _flatten_into(target, required_file)


def _download_with_service_account(folder_id: str, sa_json: str, target: Path) -> None:
    """sa_json 이 비어 있으면 Application Default Credentials(ADC)를 사용한다.

    Cloud Run/GCE 에서는 런타임 서비스 계정으로 ADC 가 자동 제공되므로, 키
    파일 없이 폴더를 그 서비스 계정 이메일과 공유하기만 하면 된다(키 유출 0).
    하위 폴더(checkpoint-*)는 건너뛰고 평면 파일만 받는다.
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
    page_token = None
    files = []
    while True:
        resp = service.files().list(
            q=query, fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=1000, pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    if not files:
        raise RuntimeError(
            f"폴더가 비었거나 서비스 계정에 공유되지 않았습니다 (folder_id={folder_id})."
        )
    for f in files:
        if f["mimeType"] == "application/vnd.google-apps.folder":
            continue  # 하위 폴더(checkpoint-*)는 건너뜀
        dest = target / f["name"]
        if dest.exists() and f.get("size") and dest.stat().st_size == int(f["size"]):
            log.info("이미 존재(크기 일치), 건너뜀: %s", f["name"])
            continue
        size_mb = (int(f["size"]) / 1e6) if f.get("size") else 0
        log.info("다운로드: %s (%.1f MB)", f["name"], size_mb)
        request = service.files().get_media(fileId=f["id"])
        with io.FileIO(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()


def download_drive_folder(folder_id: str, target_dir, required_file: str):
    """Drive 폴더를 target_dir 로 내려받는다. required_file 이 이미 있으면 생략.

    환경변수:
      CSS_LLM_GDRIVE_SA_JSON  — 서비스 계정 키 파일 경로(설정 시 비공개)
      CSS_LLM_GDRIVE_USE_ADC  — "1" 이면 런타임 SA(ADC)로 비공개 다운로드
      (둘 다 없으면 gdown 공개 링크 방식)
    """
    target = Path(target_dir)
    sa_json = os.environ.get("CSS_LLM_GDRIVE_SA_JSON", "")
    use_adc = os.environ.get("CSS_LLM_GDRIVE_USE_ADC", "0") == "1"
    sentinel = target / COMPLETE_SENTINEL

    # 완료 센티넬이 있을 때만 '받기 완료'로 인정한다(부분 다운로드 오인 방지).
    if sentinel.exists():
        log.info("이미 완료(센티넬 존재): %s (다운로드 생략)", target)
        return target
    if not folder_id:
        raise RuntimeError("Drive 폴더 ID 가 비어 있습니다.")

    # 센티넬이 없으면(최초 또는 중단된 다운로드) 다시 받는다. 서비스계정 경로는
    # 파일별 크기 검증으로 누락/절단된 파일만 재다운로드(resume)하므로, 중단 후
    # 재시도가 자가복구된다.
    if sa_json or use_adc:
        log.info("서비스 계정 방식(비공개)으로 다운로드합니다.")
        _download_with_service_account(folder_id, sa_json, target)
    else:
        log.info("gdown(공개 링크) 방식으로 다운로드합니다.")
        _download_with_gdown(folder_id, target, required_file)

    if not (target / required_file).exists():
        raise RuntimeError(
            f"다운로드는 끝났지만 {required_file} 를 찾지 못했습니다. "
            f"폴더 ID/공유 설정을 확인하세요: {target}"
        )
    # 모든 파일 수신 확인 후에만 완료 센티넬 기록(이 뒤로는 스킵된다).
    sentinel.write_text("ok\n", encoding="utf-8")
    log.info("다운로드 완료 → %s (센티넬 기록)", target)
    return target


def download_adapter_from_drive(target_dir) -> Path:
    """LoRA 어댑터를 CSS_LLM_DRIVE_FOLDER_ID 폴더에서 내려받는다."""
    folder_id = os.environ.get("CSS_LLM_DRIVE_FOLDER_ID", "")
    if not folder_id:
        raise RuntimeError("CSS_LLM_DRIVE_FOLDER_ID 가 설정되지 않았습니다.")
    return download_drive_folder(folder_id, target_dir, ADAPTER_REQUIRED_FILE)


def download_base_from_drive(target_dir) -> Path:
    """베이스 모델(Qwen3-14B)을 CSS_LLM_BASE_DRIVE_FOLDER_ID 폴더에서 내려받는다.

    HF Hub 429 우회용. 이 환경변수가 없으면 호출하지 않는다(HF 에서 받음).
    """
    folder_id = os.environ.get("CSS_LLM_BASE_DRIVE_FOLDER_ID", "")
    if not folder_id:
        raise RuntimeError("CSS_LLM_BASE_DRIVE_FOLDER_ID 가 설정되지 않았습니다.")
    return download_drive_folder(folder_id, target_dir, BASE_REQUIRED_FILE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.utils.config import ARTIFACTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true",
                        help="어댑터 대신 베이스 모델을 받는다.")
    parser.add_argument("--target", default=None, help="저장할 로컬 경로.")
    args = parser.parse_args()

    if args.base:
        target = args.target or str(ARTIFACTS_DIR / "qwen3_base")
        download_base_from_drive(target)
    else:
        target = args.target or str(ARTIFACTS_DIR / "qwen3_lora")
        download_adapter_from_drive(target)


if __name__ == "__main__":
    main()
