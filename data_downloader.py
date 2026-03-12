"""
data_downloader.py
──────────────────
Downloads the NIH ChestX-ray14 dataset sample (5,606 images, ~1.2 GB).
This is the public domain dataset used to build the visual_db.index.

The NIH dataset contains 14 disease labels per image in the CSV metadata:
  Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
  Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
  Pleural_Thickening, Hernia  (plus "No Finding")

Usage:
    python data_downloader.py --output_dir ./data
"""

import os
import sys
import time
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── NIH ChestX-ray14 public download URLs ─────────────────────────────────────
# Source: https://nihcc.app.box.com/v/ChestXray-NIHCC
# The NIH provides 12 batch ZIPs + 1 metadata CSV.
# We use only the FIRST batch (images_001.tar.gz → ~1.1 GB, 4,999 images)
# for a fast bootstrap.  Add more batches for larger gallery.

NIH_METADATA_URL = (
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/"
    "master/metadata.csv"  # placeholder – real URL below
)

# Real NIH metadata (hosted on Kaggle mirror for convenience)
NIH_KAGGLE_METADATA = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/torchxrayvision/data_dicts/nih_chest_xray_dict.json"

# ── Open-I (Indiana University) – ALWAYS freely available, no login ───────────
# 7,470 frontal X-rays  ~900 MB
OPENI_BASE = "https://openi.nlm.nih.gov/imgs/collections/"
OPENI_ARCHIVE = "NLMCXR_png.tgz"  # full archive
OPENI_METADATA_URL = "https://openi.nlm.nih.gov/api/search?q=&it=x&m=1&n=500"

# ── Lightweight fallback: Kaggle chest-xray-pneumonia (1.15 GB) ───────────────
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Requires kaggle CLI auth token.

SUPPORTED_SOURCES = ["openi", "nih_sample", "local"]


def download_with_progress(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Stream-download a file with a tqdm progress bar."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=dest_path.name, ncols=80
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def download_openi(output_dir: Path) -> Path:
    """
    Download Open-I Indiana University chest X-ray PNG collection.
    Returns the directory containing .png images.
    """
    import tarfile

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / OPENI_ARCHIVE
    images_dir = output_dir / "openi_images"

    if images_dir.exists() and any(images_dir.glob("*.png")):
        print(f"[SKIP] Open-I images already present at {images_dir}")
        return images_dir

    print("=" * 60)
    print("Downloading Open-I Indiana X-ray dataset (~900 MB)...")
    print("Source: National Library of Medicine (public domain)")
    print("=" * 60)

    url = OPENI_BASE + OPENI_ARCHIVE
    if not download_with_progress(url, archive_path):
        raise RuntimeError("Failed to download Open-I archive.")

    print(f"Extracting to {images_dir}...")
    images_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=images_dir)

    archive_path.unlink()  # free disk space
    print(f"[OK] Open-I images extracted → {images_dir}")
    return images_dir


def download_nih_sample(output_dir: Path, max_images: int = 5000) -> Path:
    """
    Download NIH ChestX-ray14 batch_01 (~4,999 images, ~1.1 GB).
    Uses direct Box.com links published by NIH.
    """
    import tarfile

    NIH_BATCH1_URL = (
        "https://nihcc.box.com/shared/static/"
        "vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "nih_images_001.tar.gz"
    images_dir = output_dir / "nih_images"

    if images_dir.exists() and any(images_dir.glob("*.png")):
        print(f"[SKIP] NIH images already present at {images_dir}")
        return images_dir

    print("=" * 60)
    print("Downloading NIH ChestX-ray14 Batch 1 (~1.1 GB)...")
    print("Source: NIH Clinical Center (CC0 license)")
    print("=" * 60)

    if not download_with_progress(NIH_BATCH1_URL, archive_path):
        raise RuntimeError(
            "Failed to download NIH batch. "
            "Try manual download from: https://nihcc.app.box.com/v/ChestXray-NIHCC"
        )

    print(f"Extracting to {images_dir}...")
    images_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()[:max_images]
        tar.extractall(path=images_dir, members=members)

    archive_path.unlink()
    print(f"[OK] NIH images extracted → {images_dir}")
    return images_dir


def download_nih_metadata(output_dir: Path) -> Path:
    """Download the NIH ChestX-ray14 labels CSV."""
    META_URL = (
        "https://raw.githubusercontent.com/mlmed/torchxrayvision/"
        "master/tests/test_data/nih_data_entry_small.csv"
    )
    # Full metadata (108,948 rows):
    FULL_META_URL = (
        "https://raw.githubusercontent.com/ieee8023/chexnet-dataset/"
        "master/Data_Entry_2017.csv"
    )
    dest = output_dir / "nih_metadata.csv"
    if dest.exists():
        return dest
    print("Downloading NIH metadata CSV...")
    download_with_progress(FULL_META_URL, dest)
    return dest


def scan_local_images(image_dir: Path) -> list[Path]:
    """Return all PNG/JPG images in a directory (recursive)."""
    extensions = {".png", ".jpg", ".jpeg"}
    images = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in extensions
    ]
    print(f"[SCAN] Found {len(images):,} images in {image_dir}")
    return images


def build_metadata_csv(
    image_dir: Path,
    nih_csv_path: Path | None,
    output_path: Path
) -> pd.DataFrame:
    """
    Build a unified metadata CSV:
        filename | filepath | labels | source
    Works whether NIH labels CSV is available or not.
    """
    images = scan_local_images(image_dir)

    rows = []
    label_lookup = {}

    if nih_csv_path and nih_csv_path.exists():
        df_nih = pd.read_csv(nih_csv_path)
        # NIH CSV cols: Image Index, Finding Labels, Patient ID, ...
        for _, row in df_nih.iterrows():
            label_lookup[row["Image Index"]] = row["Finding Labels"]

    for img_path in images:
        fname = img_path.name
        labels = label_lookup.get(fname, "Unknown")
        rows.append({
            "filename": fname,
            "filepath": str(img_path.resolve()),
            "labels": labels,
            "source": "NIH" if label_lookup else "Unknown",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[OK] Metadata saved → {output_path}  ({len(df):,} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download chest X-ray dataset for gallery builder"
    )
    parser.add_argument(
        "--source", choices=SUPPORTED_SOURCES, default="openi",
        help="Dataset source (default: openi – no login required)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./data"),
        help="Directory to save images and metadata"
    )
    parser.add_argument(
        "--local_dir", type=Path, default=None,
        help="Path to existing local image folder (use with --source local)"
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "openi":
        images_dir = download_openi(output_dir)
    elif args.source == "nih_sample":
        images_dir = download_nih_sample(output_dir)
        nih_meta = download_nih_metadata(output_dir)
        build_metadata_csv(images_dir, nih_meta, output_dir / "metadata.csv")
        return
    elif args.source == "local":
        if not args.local_dir:
            print("[ERROR] --local_dir is required when --source=local")
            sys.exit(1)
        images_dir = args.local_dir.resolve()
    else:
        print(f"[ERROR] Unknown source: {args.source}")
        sys.exit(1)

    build_metadata_csv(images_dir, None, output_dir / "metadata.csv")
    print("\n✅ Dataset ready. Next step:")
    print(f"   python gallery_builder.py --image_dir {images_dir} --output_dir ./index")


if __name__ == "__main__":
    main()
