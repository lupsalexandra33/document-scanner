"""
    python evaluate_docxpand.py --detection-only     geometry only, fast
    python evaluate_docxpand.py --limit 20      full pipeline with OCR
"""
import argparse
import glob
import json
import os
import time
import unicodedata
from collections import defaultdict

import cv2
import numpy as np

import docxpand
import preprocessing
import scanner


def normalise(text):
    # strip accents, case, punctuation and spacing before comparing
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    keep = [c.lower() for c in text if c.isalnum()]
    return "".join(keep) or None


def quad_iou(quad_a, quad_b, image_shape):
    # intersection over union of two quadrilaterals, computed on masks
    height, width = image_shape[:2]
    mask_a = np.zeros((height, width), np.uint8)
    mask_b = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask_a, [np.int32(quad_a)], 1)
    cv2.fillPoly(mask_b, [np.int32(quad_b)], 1)
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union if union else 0.0


def compare_fields(extracted, truth):
    # classify each ground-truth field as correct, wrong or missing
    correct = wrong = missing = 0
    per_field = {}
    for name, true_value in truth.items():
        if true_value is None:
            continue                       # not printed on this document
        got = extracted.get(name)
        if got is None:
            missing += 1
            per_field[name] = "missing"
        elif normalise(got) == normalise(true_value):
            correct += 1
            per_field[name] = "correct"
        else:
            wrong += 1
            per_field[name] = "wrong"
    return correct, wrong, missing, per_field


def evaluate(files, ground_truth, with_ocr, iou_threshold=0.5):
    rows = []
    field_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "missing": 0})
    extract = None
    if with_ocr:
        import extract as extract      # slow import: loads the OCR model

    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue

        start = time.time()
        quad, method = preprocessing.detect_document_improved(image)
        detect_time = time.time() - start

        true_quad = ground_truth.quad(path, image.shape)
        iou = None
        if quad is not None and true_quad is not None:
            iou = quad_iou(scanner.order_points(quad.reshape(4, 2)),
                           np.array(true_quad, np.float32), image.shape)

        row = {
            "file": os.path.basename(path),
            "class": ground_truth.document_class(path),
            "detected": quad is not None,
            "method": method,
            "iou": round(iou, 3) if iou is not None else None,
            "detect_seconds": round(detect_time, 3),
        }

        if with_ocr:
            start = time.time()
            try:
                result = extract.process_document(path)
                row["ocr_seconds"] = round(time.time() - start, 2)
                correct, wrong, missing, per_field = compare_fields(
                    result["fields"], ground_truth.fields(path))
                row.update(correct=correct, wrong=wrong, missing=missing)
                for name, verdict in per_field.items():
                    field_stats[name][verdict] += 1
            except Exception as exc:
                row["error"] = str(exc)

        rows.append(row)

    return rows, field_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/docxpand/DocXPand-25k/documents")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--detection-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    labels_path = args.labels or docxpand.find_labels()
    if not labels_path:
        print("DocXPand-25k.json not found — download it with:")
        print("  python download_docxpand.py --labels")
        return

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no images under {args.data}/")
        return

    print(f"loading ground truth from {labels_path}...")
    ground_truth = docxpand.GroundTruth(labels_path)
    print(f"evaluating {len(files)} images\n")

    rows, field_stats = evaluate(files, ground_truth, not args.detection_only,
                                 args.iou_threshold)

    # localisation
    detected = [r for r in rows if r["detected"]]
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    good = [i for i in ious if i >= args.iou_threshold]

    print("=== document localisation ===")
    print(f"detected:            {len(detected)}/{len(rows)}")
    if ious:
        print(f"mean IoU:            {sum(ious)/len(ious):.3f}")
        print(f"IoU >= {args.iou_threshold}:          {len(good)}/{len(ious)}")
    by_class = defaultdict(lambda: [0, 0])
    for r in rows:
        by_class[r["class"]][1] += 1
        if r["detected"]:
            by_class[r["class"]][0] += 1
    for cls, (ok, total) in sorted(by_class.items()):
        print(f"  {cls:16s} {ok}/{total}")

    failures = [r["file"] for r in rows if not r["detected"]]
    if failures:
        print(f"\nfailure cases ({len(failures)}): {', '.join(failures[:6])}"
              f"{' ...' if len(failures) > 6 else ''}")

    avg_detect = sum(r["detect_seconds"] for r in rows) / len(rows)
    print(f"\naverage detection time: {avg_detect*1000:.0f} ms")

    # field extraction
    if not args.detection_only:
        total_c = sum(r.get("correct", 0) for r in rows)
        total_w = sum(r.get("wrong", 0) for r in rows)
        total_m = sum(r.get("missing", 0) for r in rows)
        total = total_c + total_w + total_m

        print("\n=== field extraction ===")
        if total:
            print(f"correct: {total_c}/{total} ({100*total_c/total:.0f}%)")
            print(f"wrong:   {total_w}/{total} ({100*total_w/total:.0f}%)")
            print(f"missing: {total_m}/{total} ({100*total_m/total:.0f}%)")

        print(f"\n{'field':20s} {'correct':>8s} {'wrong':>7s} {'missing':>8s}")
        for name, stats in sorted(field_stats.items()):
            print(f"{name:20s} {stats['correct']:>8d} {stats['wrong']:>7d} "
                  f"{stats['missing']:>8d}")

        ocr_times = [r["ocr_seconds"] for r in rows if "ocr_seconds" in r]
        if ocr_times:
            print(f"\naverage time per document: {sum(ocr_times)/len(ocr_times):.1f} s")

    with open("docxpand_evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "field_stats": {k: dict(v) for k, v in field_stats.items()}},
                  f, indent=2, ensure_ascii=False)
    print("\nsaved to docxpand_evaluation.json")


if __name__ == "__main__":
    main()