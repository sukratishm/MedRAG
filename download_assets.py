"""
download_assets.py
------------------
Downloads index/ and image assets from Google Drive into /var/data.

Env vars:
  GDRIVE_INDEX_URL  - share link or direct download url for a zip/tar of index/
  GDRIVE_IMAGES_URL - share link or direct download url for a zip/tar of images/
  DATA_DIR          - base path (default: /var/data)
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download


BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    gdown.download(url, str(dest), quiet=False)
    return dest


def _extract(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    elif tarfile.is_tarfile(archive) or archive.name.endswith((".tgz", ".tar.gz", ".gz")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_data_dir() -> Path:
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    for candidate in (Path("/var/data"), Path("/tmp/medrag_data")):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return Path("/tmp/medrag_data").resolve()


def _prefetch_biomedclip() -> None:
    cache_dir = Path(os.getenv("HF_HOME", "/tmp/hf_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BIOMEDCLIP_REPO,
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    print(f"BiomedCLIP cached in {cache_dir}")


def main():
    data_dir = _pick_data_dir()
    index_dir = data_dir / "index"
    images_dir = data_dir / "images"

    index_url = os.getenv("GDRIVE_INDEX_URL", "")
    images_url = os.getenv("GDRIVE_IMAGES_URL", "")

    _ensure_dir(data_dir)

    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"Index already present at {index_dir}")
    elif index_url:
        archive = data_dir / "index_archive.zip"
        archive = _download(index_url, archive)
        _extract(archive, index_dir)
        print(f"Index extracted to {index_dir}")
    else:
        print("GDRIVE_INDEX_URL not set; index not downloaded.")

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Images already present at {images_dir}")
    elif images_url:
        archive = data_dir / "images_archive.zip"
        archive = _download(images_url, archive)
        _extract(archive, images_dir)
        print(f"Images extracted to {images_dir}")
    else:
        print("GDRIVE_IMAGES_URL not set; images not downloaded.")

    # cleanup
    for f in [data_dir / "index_archive.zip", data_dir / "images_archive.zip"]:
        if f.exists():
            try:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
            except Exception:
                pass

    if os.getenv("PREFETCH_MODEL", "1") == "1":
        _prefetch_biomedclip()


if __name__ == "__main__":
    main()
