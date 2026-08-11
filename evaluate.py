"""
two modes:
    python evaluate.py --detection-only     fast, geometry only, no OCR
    python evaluate.py      full pipeline including OCR

the results are printed as a table and written to evaluation_results.json.
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import cv2

import scanner
import preprocessing


def condition_of(path):
    # MIDV encodes the capture condition in the filename: TS38_01.tif => 'TS'.
    return os.path.basename(path)[:2]


def evaluate_detection(files):
    # Compare the baseline detector against the improved one, per condition
    stats = defaultdict(lambda: {"total": 0, "baseline": 0, "improved": 0,
                                 "contour": 0, "minarea": 0})
    failures = []
    t_base = t_impr = 0.0

    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue
        cond = condition_of(path)
        stats[cond]["total"] += 1

        t0 = time.time()
        quad_base = scanner.detect_document(image)
        t_base += time.time() - t0

        t0 = time.time()
        quad_impr, method = preprocessing.detect_document_improved(image)
        t_impr += time.time() - t0

        if quad_base is not None:
            stats[cond]["baseline"] += 1
        if quad_impr is not None:
            stats[cond]["improved"] += 1
            stats[cond][method] += 1
        else:
            failures.append(os.path.basename(path))

    n = max(len(files), 1)
    return stats, failures, t_base / n, t_impr / n


def evaluate_full(files):
    # run the whole pipeline (OCR included) and count extracted fields
    import extract          # imported lazily: loading EasyOCR is slow

    per_image = []
    field_hits = defaultdict(int)
    total_time = 0.0

    for path in files:
        t0 = time.time()
        try:
            result = extract.process_document(path)
        except Exception as exc:                      # keep going on a bad image
            per_image.append({"file": os.path.basename(path), "error": str(exc)})
            continue
        elapsed = time.time() - t0
        total_time += elapsed

        fields = result["fields"]
        extracted = [k for k, v in fields.items() if v is not None]
        for k in extracted:
            field_hits[k] += 1

        per_image.append({
            "file": os.path.basename(path),
            "condition": condition_of(path),
            "extracted": len(extracted),
            "missing": len(fields) - len(extracted),
            "warnings": len(result["validation"]["warnings"]),
            "ocr_confidence": result["validation"]["ocr_confidence"],
            "seconds": round(elapsed, 2),
        })

    return per_image, field_hits, total_time / max(len(files), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="dataset folder")
    parser.add_argument("--detection-only", action="store_true",
                        help="skip OCR, measure geometry only (fast)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.tif"), recursive=True))
    if not files:
        print(f"no .tif images found under {args.data}/ — run download_dataset.py first")
        return

    print(f"evaluating {len(files)} images\n")
    report = {"images": len(files)}

    # detection metrics
    stats, failures, t_base, t_impr = evaluate_detection(files)

    print(f"{'cond':6s} {'n':>3s} {'baseline':>10s} {'improved':>10s}   method")
    tot_b = tot_i = tot_n = 0
    for cond in sorted(stats):
        s = stats[cond]
        tot_n += s["total"]; tot_b += s["baseline"]; tot_i += s["improved"]
        method = f"contour {s['contour']}, minarea {s['minarea']}"
        print(f"{cond:6s} {s['total']:3d} {s['baseline']:>10d} {s['improved']:>10d}   {method}")
    print(f"{'ALL':6s} {tot_n:3d} {tot_b:>10d} {tot_i:>10d}")
    print(f"\ndetection rate: baseline {tot_b}/{tot_n} ({100*tot_b/tot_n:.0f}%), "
          f"improved {tot_i}/{tot_n} ({100*tot_i/tot_n:.0f}%)")
    print(f"average detection time: baseline {t_base*1000:.0f} ms, improved {t_impr*1000:.0f} ms")
    if failures:
        print(f"\nfailure cases ({len(failures)}): {', '.join(failures)}")

    report["detection"] = {
        "per_condition": {c: dict(v) for c, v in stats.items()},
        "baseline_rate": round(tot_b / tot_n, 3),
        "improved_rate": round(tot_i / tot_n, 3),
        "baseline_ms": round(t_base * 1000),
        "improved_ms": round(t_impr * 1000),
        "failures": failures,
    }

    # field extraction metrics
    if not args.detection_only:
        print("\nrunning full pipeline with OCR (this is slow on CPU)...\n")
        per_image, field_hits, avg_t = evaluate_full(files)

        print(f"{'file':16s} {'cond':5s} {'got':>4s} {'miss':>5s} {'warn':>5s} {'conf':>6s} {'sec':>6s}")
        for r in per_image:
            if "error" in r:
                print(f"{r['file']:16s} ERROR: {r['error']}")
                continue
            conf = f"{r['ocr_confidence']:.2f}" if r["ocr_confidence"] is not None else "-"
            print(f"{r['file']:16s} {r['condition']:5s} {r['extracted']:>4d} "
                  f"{r['missing']:>5d} {r['warnings']:>5d} {conf:>6s} {r['seconds']:>6.1f}")

        print("\nfields extracted, across all images:")
        for field, count in sorted(field_hits.items(), key=lambda kv: -kv[1]):
            print(f"  {field:18s} {count}/{len(files)}")
        print(f"\naverage runtime per image: {avg_t:.1f} s")

        report["extraction"] = {
            "per_image": per_image,
            "field_hits": dict(field_hits),
            "avg_seconds": round(avg_t, 2),
        }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nsaved to evaluation_results.json")


if __name__ == "__main__":
    main()