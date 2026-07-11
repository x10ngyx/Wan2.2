#!/usr/bin/env python3
"""Pack the normalized HF staging dataset into OSS transfer archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


THRESHOLD_KEYS = [
    "threshold_0p10",
    "threshold_0p15",
    "threshold_0p20",
    "threshold_0p25",
    "threshold_0p30",
    "threshold_0p40",
    "threshold_0p50",
    "threshold_0p60",
    "threshold_0p70",
    "threshold_0p80",
]


UPLOAD_GUIDE = """# Upload Restored Dataset to Hugging Face

This package was produced as an OSS transfer bundle. It is not meant to be
uploaded to Hugging Face as-is. First restore the Hugging Face-ready dataset
layout from the archives, verify it, then upload the restored directory.

## 1. Download From OSS

Example:

```bash
ossutil cp -r oss://<bucket>/<prefix>/wan22_openvid100_seacache_trace_oss_bundle/ \\
  /data/wan22_openvid100_seacache_trace_oss_bundle/
```

Use your actual bucket, endpoint, and credential configuration.

## 2. Verify Archive Checksums

```bash
cd /data/wan22_openvid100_seacache_trace_oss_bundle
sha256sum -c checksums/archive_sha256s.txt
```

Every line should report `OK`.

## 3. Restore the Hugging Face Dataset Directory

Choose a location with enough free space for the restored dataset.

```bash
mkdir -p /data/hf_restore/wan22_openvid100_seacache_trace
for archive in archives/*.tar; do
  tar -xf "$archive" -C /data/hf_restore/wan22_openvid100_seacache_trace
done
```

The restored directory should contain:

```text
README.md
dataset_manifest.json
data/
media/
traces/
artifacts/
metadata/
checksums/
scripts/
```

## 4. Verify Restored Dataset

```bash
cd /data/hf_restore/wan22_openvid100_seacache_trace
python scripts/verify_dataset.py .
```

This checks the expected 100 prompts, 1000 candidate rows, 50,000 training rows,
and the presence of key media/trace paths referenced by the index tables.

## 5. Install Hugging Face CLI

Use a machine with good network access to Hugging Face.

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login
```

For high-throughput upload on newer `huggingface_hub` versions:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

## 6. Create the Dataset Repo

Replace `<namespace>/<repo_name>` with the target repository.

```bash
hf repo create <namespace>/<repo_name> --type dataset
```

If the repo already exists, this command may fail harmlessly; continue to the
upload step.

## 7. Upload

```bash
cd /data/hf_restore/wan22_openvid100_seacache_trace
hf upload <namespace>/<repo_name> . . --repo-type dataset
```

If the upload is interrupted, rerun the same command. The Hub client skips
files already uploaded.

## 8. Post-Upload Sanity Check

After upload, confirm that the repo page shows:

- `README.md`
- `data/train-00000-of-00010.parquet` through `train-00009-of-00010.parquet`
- `data/candidates.parquet`
- `data/prompts.parquet`
- `media/`
- `traces/`
- `artifacts/`

Then download a small subset or clone the repo on a clean machine and run:

```bash
python scripts/verify_dataset.py .
```
"""


VERIFY_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.dataset_root)

    manifest = json.loads((root / "dataset_manifest.json").read_text())
    prompts = pd.read_parquet(root / "data" / "prompts.parquet")
    candidates = pd.read_parquet(root / "data" / "candidates.parquet")
    train_files = sorted((root / "data").glob("train-*.parquet"))
    train_rows = sum(len(pd.read_parquet(path, columns=["row_id"])) for path in train_files)

    checks = {
        "prompts": len(prompts) == 100,
        "candidates": len(candidates) == 1000,
        "train_rows": train_rows == 50000,
        "manifest_train_rows": manifest.get("training_rows") == 50000,
        "baseline_video_prompt_001": (root / "media" / "baseline" / "prompt_001.mp4").exists(),
        "seacache_video_prompt_001_0p10": (root / "media" / "seacache" / "threshold_0p10" / "prompt_001.mp4").exists(),
        "trace_prompt_001_0p10_step_000": (root / "traces" / "seacache" / "threshold_0p10" / "prompt_001" / "step_000.pt").exists(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print(json.dumps({"checks": checks, "train_files": len(train_files), "train_rows": train_rows}, indent=2))
    if failed:
        raise SystemExit(f"dataset verification failed: {failed}")


if __name__ == "__main__":
    main()
'''


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_tar(staging_root: Path, archive_path: Path, members: list[str]) -> None:
    existing = [m for m in members if (staging_root / m).exists()]
    if not existing:
        print(f"skip empty archive {archive_path.name}")
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "--dereference", "-cf", str(archive_path), "-C", str(staging_root), *existing]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", default="/hy-tmp/hf_staging/wan22_openvid100_seacache_trace")
    parser.add_argument("--oss-bundle-root", default="")
    args = parser.parse_args()

    staging_root = Path(args.staging_root).resolve()
    if not staging_root.exists():
        raise SystemExit(f"missing staging root: {staging_root}")

    scripts_dir = staging_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "verify_dataset.py").write_text(VERIFY_SCRIPT, encoding="utf-8")
    os.chmod(scripts_dir / "verify_dataset.py", 0o755)
    (staging_root / "HOW_TO_UPLOAD_TO_HUGGINGFACE.md").write_text(UPLOAD_GUIDE, encoding="utf-8")

    if args.oss_bundle_root:
        bundle_root = Path(args.oss_bundle_root).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bundle_root = Path(f"/hy-tmp/oss_upload/wan22_openvid100_seacache_trace_{stamp}")
    archives_dir = bundle_root / "archives"
    checksums_dir = bundle_root / "checksums"
    archives_dir.mkdir(parents=True, exist_ok=True)
    checksums_dir.mkdir(parents=True, exist_ok=True)

    archive_plan: list[tuple[str, list[str]]] = [
        (
            "repo_metadata_and_indexes.tar",
            [
                "README.md",
                "HOW_TO_UPLOAD_TO_HUGGINGFACE.md",
                "dataset_manifest.json",
                "data",
                "metadata",
                "checksums",
                "scripts",
            ],
        ),
        ("media_baseline.tar", ["media/baseline"]),
        ("traces_baseline.tar", ["traces/baseline"]),
        ("artifacts_logs.tar", ["artifacts/logs"]),
        ("artifacts_commands.tar", ["artifacts/commands"]),
        ("artifacts_ffprobe.tar", ["artifacts/ffprobe"]),
        ("artifacts_psnr.tar", ["artifacts/psnr"]),
    ]
    for key in THRESHOLD_KEYS:
        archive_plan.append((f"media_seacache_{key}.tar", [f"media/seacache/{key}"]))
    for key in THRESHOLD_KEYS:
        archive_plan.append((f"traces_seacache_{key}.tar", [f"traces/seacache/{key}"]))

    archive_records = []
    for name, members in archive_plan:
        archive = archives_dir / name
        run_tar(staging_root, archive, members)
        if archive.exists():
            archive_records.append(
                {
                    "archive": f"archives/{name}",
                    "bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive),
                    "members": members,
                }
            )

    lines = [f"{r['sha256']}  {r['archive']}\n" for r in archive_records]
    (checksums_dir / "archive_sha256s.txt").write_text("".join(lines), encoding="utf-8")
    (bundle_root / "HOW_TO_UPLOAD_TO_HUGGINGFACE.md").write_text(UPLOAD_GUIDE, encoding="utf-8")
    (bundle_root / "archive_manifest.json").write_text(json.dumps(archive_records, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"bundle_root": str(bundle_root), "archives": archive_records}, indent=2))


if __name__ == "__main__":
    main()
