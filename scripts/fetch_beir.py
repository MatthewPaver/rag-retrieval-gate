#!/usr/bin/env python3
"""Fetch BEIR SciFact locally with checksum and licence gating. Data is never committed.

Primary source: the BEIR zip on the UKP server (MD5 published in the BEIR catalogue).
Fallback (--source huggingface): the BeIR/scifact parquet mirror on Hugging Face, pinned by
SHA-256 and converted to the same corpus.jsonl / queries.jsonl / qrels/test.tsv layout.
The fallback needs `pip install pyarrow`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_CATALOGUE = "https://github.com/beir-cellar/beir/wiki/Datasets-available"
UKP_ZIP = {
    "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "md5": "5f7d1de60b170fc8027bb7898e2efca1",
}
HF_BASE = "https://huggingface.co/datasets/BeIR"
HF_FILES = {
    "corpus.parquet": (
        f"{HF_BASE}/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
        "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
    ),
    "queries.parquet": (
        f"{HF_BASE}/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
        "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
    ),
    "test.tsv": (
        f"{HF_BASE}/scifact-qrels/resolve/main/test.tsv",
        "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    ),
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, target: Path, expected: str, algorithm: str) -> None:
    urllib.request.urlretrieve(url, target)
    actual = digest(target, algorithm)
    if actual != expected:
        target.unlink()
        raise SystemExit(f"Checksum mismatch for {url}: expected {expected}, got {actual}")


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def fetch_ukp(output_root: Path, archive: Path | None) -> str:
    archive_path = output_root / "scifact.zip"
    if archive:
        shutil.copyfile(archive, archive_path)
        if digest(archive_path, "md5") != UKP_ZIP["md5"]:
            raise SystemExit(f"Checksum mismatch for {archive}")
    else:
        download(UKP_ZIP["url"], archive_path, UKP_ZIP["md5"], "md5")
    with zipfile.ZipFile(archive_path) as handle:
        safe_extract(handle, output_root)
    return UKP_ZIP["url"]


def fetch_huggingface(output_root: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("--source huggingface needs: pip install pyarrow") from exc
    dataset_dir = output_root / "scifact"
    staging = output_root / "scifact-hf"
    (dataset_dir / "qrels").mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in HF_FILES.items():
        download(url, staging / name, sha256, "sha256")
    for name in ("corpus", "queries"):
        rows = pq.read_table(staging / f"{name}.parquet").to_pylist()
        with (dataset_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({key: row.get(key, "") for key in ("_id", "title", "text")}) + "\n")
    shutil.copyfile(staging / "test.tsv", dataset_dir / "qrels" / "test.tsv")
    return f"{HF_BASE}/scifact (+ scifact-qrels)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--acknowledge-licence",
        action="store_true",
        help=f"confirm you reviewed the SciFact licence listed in {DATASET_CATALOGUE}",
    )
    parser.add_argument("--source", choices=["ukp", "huggingface"], default="ukp")
    parser.add_argument("--output-root", type=Path, default=Path("data/beir"))
    parser.add_argument("--archive", type=Path, help="use a pre-downloaded UKP scifact.zip")
    args = parser.parse_args()
    if not args.acknowledge_licence:
        raise SystemExit(
            "Refusing to fetch without --acknowledge-licence. "
            f"Review {DATASET_CATALOGUE} and the SciFact licence first."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    source = (
        fetch_huggingface(args.output_root)
        if args.source == "huggingface"
        else fetch_ukp(args.output_root, args.archive)
    )
    dataset_dir = args.output_root / "scifact"
    required = [dataset_dir / "corpus.jsonl", dataset_dir / "queries.jsonl", dataset_dir / "qrels" / "test.tsv"]
    if not all(path.exists() for path in required):
        raise SystemExit(f"Dataset is incomplete: {dataset_dir}")
    provenance = {
        "dataset": "scifact",
        "source": source,
        "catalogue_and_licence_url": DATASET_CATALOGUE,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "redistribution": "not redistributed; downloaded locally under the dataset's own terms",
    }
    (dataset_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
