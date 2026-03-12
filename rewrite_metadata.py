"""
rewrite_metadata.py
-------------------
Utility to rewrite metadata.json filepaths for deployment.

Example:
  python rewrite_metadata.py \
    --index_dir ./index \
    --from_prefix "/Users/you/MedRAG/data/train" \
    --to_prefix "/var/data/images"
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rewrite metadata.json filepaths")
    parser.add_argument("--index_dir", type=Path, default=Path("./index"))
    parser.add_argument("--from_prefix", required=True)
    parser.add_argument("--to_prefix", required=True)
    args = parser.parse_args()

    meta_path = args.index_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    data = json.loads(meta_path.read_text())
    updated = 0

    for _, entry in data.items():
        fp = entry.get("filepath", "")
        if fp.startswith(args.from_prefix):
            entry["filepath"] = fp.replace(args.from_prefix, args.to_prefix, 1)
            updated += 1

    meta_path.write_text(json.dumps(data, indent=2))
    print(f"Rewrote {updated} filepaths in {meta_path}")


if __name__ == "__main__":
    main()
