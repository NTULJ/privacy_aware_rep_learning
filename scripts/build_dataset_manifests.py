from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import extract_qwen_vl_features as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a master manifest and round-robin shard manifests for feature extraction.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("auto", "stanford40", "lfw", "generic"), default="auto")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError(f"shard-count must be positive, got {args.shard_count}")

    dataset_root = args.dataset_root.resolve()
    samples = base.unique_samples(base.scan_dataset(dataset_root, args.dataset_name))
    records = [asdict(sample) for sample in samples]

    args.master_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.master_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    shards = [[] for _ in range(args.shard_count)]
    for idx, record in enumerate(records):
        shards[idx % args.shard_count].append(record)

    args.shard_manifest_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx, shard_records in enumerate(shards):
        shard_path = args.shard_manifest_dir / f"shard_{shard_idx:02d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for record in shard_records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"prepared {len(records)} samples into {args.shard_count} shards at {args.shard_manifest_dir}")


if __name__ == "__main__":
    main()
