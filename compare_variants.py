"""
    python compare_variants.py      uses a small default set
    python compare_variants.py --data data --limit 6
"""
import argparse
import glob
import json
import os
import re
import time

import cv2

import preprocessing
import extract


def score_variant(image, quad, variant_name):
    # run OCR on one preprocessing variant and score the result
    prepared = preprocessing.PREPROCESSING_VARIANTS[variant_name](image, quad)

    t0 = time.time()
    results = extract.reader.readtext(prepared)
    elapsed = time.time() - t0

    confident = [r for r in results if r[2] >= 0.5]
    mean_conf = sum(r[2] for r in results) / len(results) if results else 0.0

    # how many target patterns the rules can still find in this variant
    text_blob = " ".join(r[1] for r in results)
    hits = 0
    hits += len(set(re.findall(extract.DATE_PATTERN, text_blob)))
    hits += 1 if re.search(extract.DOCNUM_PATTERN, text_blob) else 0
    hits += 1 if re.search(extract.CNP_PATTERN, text_blob) else 0

    return {
        "regions": len(results),
        "confident_regions": len(confident),
        "mean_confidence": round(mean_conf, 3),
        "pattern_hits": hits,
        "seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--limit", type=int, default=4,
                        help="how many images to test (OCR on CPU is slow)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.tif"), recursive=True))
    files = files[:args.limit]
    if not files:
        print(f"no images under {args.data}/ — run download_dataset.py first")
        return

    print(f"comparing {len(preprocessing.PREPROCESSING_VARIANTS)} variants "
          f"on {len(files)} images\n")

    totals = {name: {"regions": 0, "confident_regions": 0, "pattern_hits": 0,
                     "mean_confidence": 0.0, "seconds": 0.0}
              for name in preprocessing.PREPROCESSING_VARIANTS}

    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        quad, _ = preprocessing.detect_document_improved(image)
        print(f"--- {os.path.basename(path)} "
              f"({'detected' if quad is not None else 'NOT detected'}) ---")
        print(f"{'variant':14s} {'regions':>8s} {'conf>=.5':>9s} "
              f"{'mean conf':>10s} {'hits':>5s} {'sec':>6s}")

        for name in preprocessing.PREPROCESSING_VARIANTS:
            s = score_variant(image, quad, name)
            for k in totals[name]:
                totals[name][k] += s[k]
            print(f"{name:14s} {s['regions']:>8d} {s['confident_regions']:>9d} "
                  f"{s['mean_confidence']:>10.3f} {s['pattern_hits']:>5d} "
                  f"{s['seconds']:>6.1f}")
        print()

    n = len(files)
    print("=== averages across all images ===")
    print(f"{'variant':14s} {'regions':>8s} {'conf>=.5':>9s} "
          f"{'mean conf':>10s} {'hits':>5s} {'sec':>6s}")
    summary = {}
    for name, t in totals.items():
        row = {
            "regions": round(t["regions"] / n, 1),
            "confident_regions": round(t["confident_regions"] / n, 1),
            "mean_confidence": round(t["mean_confidence"] / n, 3),
            "pattern_hits": round(t["pattern_hits"] / n, 1),
            "seconds": round(t["seconds"] / n, 2),
        }
        summary[name] = row
        print(f"{name:14s} {row['regions']:>8.1f} {row['confident_regions']:>9.1f} "
              f"{row['mean_confidence']:>10.3f} {row['pattern_hits']:>5.1f} "
              f"{row['seconds']:>6.1f}")

    with open("variant_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"images": [os.path.basename(p) for p in files],
                   "averages": summary}, f, indent=2)
    print("\nsaved to variant_comparison.json")


if __name__ == "__main__":
    main()