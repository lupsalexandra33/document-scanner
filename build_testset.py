"""
    python build_testset.py                 20 images, seed 42
    python build_testset.py --size 40       larger sample
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict


def side_of(filename):
    lowered = filename.lower()
    if "front" in lowered:
        return "front"
    if "back" in lowered:
        return "back"
    return "unknown"


def build(data_dir, size, seed):
    # stratified sample: equal share per (class, side) bucket
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return [], {}

    buckets = defaultdict(list)
    for path in files:
        document_class = os.path.basename(os.path.dirname(path))
        buckets[(document_class, side_of(os.path.basename(path)))].append(path)

    rng = random.Random(seed)
    per_bucket = max(1, size // len(buckets))

    selected = []
    for key in sorted(buckets):
        pool = sorted(buckets[key])           # sort first, so the seed decides
        selected += rng.sample(pool, min(per_bucket, len(pool)))

    # if rounding left us short, top up from the remaining files at random
    if len(selected) < size:
        remaining = sorted(set(files) - set(selected))
        selected += rng.sample(remaining, min(size - len(selected), len(remaining)))

    selected.sort()
    counts = {f"{c}/{s}": len(v) for (c, s), v in sorted(buckets.items())}
    return selected, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/docxpand/DocXPand-25k/documents")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="testset.json")
    args = parser.parse_args()

    selected, available = build(args.data, args.size, args.seed)
    if not selected:
        print(f"no images under {args.data}/")
        return

    chosen = defaultdict(int)
    for path in selected:
        document_class = os.path.basename(os.path.dirname(path))
        chosen[f"{document_class}/{side_of(os.path.basename(path))}"] += 1

    print(f"selected {len(selected)} images (seed {args.seed})\n")
    print(f"{'bucket':28s} {'chosen':>7s} {'available':>10s}")
    for bucket in sorted(available):
        print(f"{bucket:28s} {chosen.get(bucket, 0):>7d} {available[bucket]:>10d}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "size": len(selected),
            "selection": "stratified random over (document class, side), "
                         "drawn before any pipeline was run",
            "buckets": dict(chosen),
            "files": selected,
        }, f, indent=2)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
