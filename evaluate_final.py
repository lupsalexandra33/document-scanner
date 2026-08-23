"""
    python evaluate_final.py --detection-only     geometry only, fast
    python evaluate_final.py --pipeline classic
    python evaluate_final.py --pipeline both      slow: Donut is ~180 s/image
"""
import argparse
import gc
import json
import os
import time
import unicodedata
from collections import defaultdict

import cv2
import numpy as np

import docxpand
import mrz as mrz_module
import preprocessing
import scanner

OUTCOMES = ("correct", "wrong", "missing", "reject")


def normalise(text):
    # trip accents, case and punctuation before comparing, those differences
    # are formatting, not extraction errors
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c.lower() for c in text if c.isalnum()) or None


def quad_iou(quad_a, quad_b, image_shape):
    # intersection over union of two quadrilaterals, computed on masks
    height, width = image_shape[:2]
    mask_a = np.zeros((height, width), np.uint8)
    mask_b = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask_a, [np.int32(quad_a)], 1)
    cv2.fillPoly(mask_b, [np.int32(quad_b)], 1)
    union = np.count_nonzero(mask_a | mask_b)
    return np.count_nonzero(mask_a & mask_b) / union if union else 0.0


def classify_fields(extracted, truth, document_detected):
    # assign one of the four outcomes to every field the document really has
    outcomes = {}
    for name, true_value in truth.items():
        if true_value is None:
            continue                       # not printed on this document
        got = extracted.get(name) if extracted else None
        if got is None:
            outcomes[name] = "missing" if document_detected else "reject"
        elif normalise(got) == normalise(true_value):
            outcomes[name] = "correct"
        else:
            outcomes[name] = "wrong"
    return outcomes


def evaluate_mrz(image_path, ocr_texts, ground_truth):
    # measure the MRZ stage end to end, separating reading from parsing
    truth_lines = ground_truth.mrz(image_path)
    if not truth_lines:
        return {"has_mrz": False}

    found = mrz_module.find_mrz_lines(ocr_texts)
    if not found:
        return {"has_mrz": True, "lines_found": False, "parsed": False}

    parsed = mrz_module.parse(found)
    if not parsed:
        return {"has_mrz": True, "lines_found": True, "parsed": False}

    # how much of the true MRZ text survived OCR, character for character
    read_correctly = sum(
        1 for a, b in zip("".join(found), "".join(truth_lines)) if a == b)
    total_chars = max(len("".join(truth_lines)), 1)

    checks = parsed.get("checks", {})
    return {
        "has_mrz": True,
        "lines_found": True,
        "parsed": True,
        "char_accuracy": round(read_correctly / total_chars, 3),
        "check_digits_passed": sum(1 for v in checks.values() if v is True),
        "check_digits_total": len([v for v in checks.values() if v is not None]),
    }


def run_pipeline(name, image_path):
    if name == "classic":
        import extract
        return extract.process_document(image_path)
    import donut_pipeline
    return donut_pipeline.process_document(image_path)


def evaluate(files, ground_truth, pipelines, detection_only):
    rows = []
    tally = {p: defaultdict(int) for p in pipelines}
    field_tally = {p: defaultdict(lambda: defaultdict(int)) for p in pipelines}
    mrz_rows = []
    ious = []

    for path in files:
        image = cv2.imread(path)
        if image is None:
            continue

        quad, method = preprocessing.detect_document_improved(image)
        true_quad = ground_truth.quad(path, image.shape)
        iou = None
        if quad is not None and true_quad is not None:
            iou = quad_iou(scanner.order_points(quad.reshape(4, 2)),
                           np.array(true_quad, np.float32), image.shape)
            ious.append(iou)

        row = {
            "file": os.path.basename(path),
            "class": ground_truth.document_class(path),
            "side": "front" if "front" in path.lower() else "back",
            "detected": quad is not None,
            "method": method,
            "iou": round(iou, 3) if iou is not None else None,
        }

        if not detection_only:
            truth = ground_truth.fields(path)
            for pipeline in pipelines:
                started = time.time()
                try:
                    result = run_pipeline(pipeline, path)
                    fields = result["fields"]
                except Exception as exc:
                    row[pipeline] = {"error": str(exc)}
                    continue
                elapsed = time.time() - started

                # Donut ignores detection, so nothing is ever rejected for it
                detected_for = True if pipeline == "donut" else (quad is not None)
                outcomes = classify_fields(fields, truth, detected_for)

                for field_name, outcome in outcomes.items():
                    tally[pipeline][outcome] += 1
                    field_tally[pipeline][field_name][outcome] += 1
                tally[pipeline]["seconds"] += elapsed

                row[pipeline] = {
                    "outcomes": outcomes,
                    "seconds": round(elapsed, 1),
                    "fields": {k: v for k, v in fields.items() if v},
                }

                if pipeline == "classic":
                    # the raw OCR text, not the extracted fields — otherwise we
                    # would be measuring the rules, not the MRZ reading itself
                    ocr_texts = result.get("ocr_texts") or []
                    mrz_result = evaluate_mrz(path, ocr_texts, ground_truth)
                    mrz_result["file"] = os.path.basename(path)
                    mrz_rows.append(mrz_result)

        rows.append(row)
        # release the decoded image before the next iteration — on CPU these
        # are large and the interpreter is slow to reclaim them on its own
        del image
        gc.collect()

    return rows, tally, field_tally, mrz_rows, ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="testset.json")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--pipeline", default="classic",
                        choices=["classic", "donut", "both"])
    parser.add_argument("--detection-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="final_evaluation.json")
    args = parser.parse_args()

    if not os.path.exists(args.testset):
        print(f"{args.testset} not found — run build_testset.py first")
        return
    with open(args.testset, encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest["files"][:args.limit] if args.limit else manifest["files"]

    labels_path = args.labels or docxpand.find_labels()
    if not labels_path:
        print("DocXPand labels not found — run download_docxpand.py --labels")
        return
    ground_truth = docxpand.GroundTruth(labels_path)

    pipelines = ["classic", "donut"] if args.pipeline == "both" else [args.pipeline]
    print(f"test set: {len(files)} images (seed {manifest['seed']})")
    print(f"pipelines: {', '.join(pipelines)}\n")

    rows, tally, field_tally, mrz_rows, ious = evaluate(
        files, ground_truth, pipelines, args.detection_only)

    # localisation
    detected = sum(1 for r in rows if r["detected"])
    print("=== document localisation ===")
    print(f"detected: {detected}/{len(rows)}")
    if ious:
        print(f"mean IoU: {sum(ious)/len(ious):.3f}")
    by_side = defaultdict(lambda: [0, 0])
    for r in rows:
        by_side[r["side"]][1] += 1
        if r["detected"]:
            by_side[r["side"]][0] += 1
    for side, (ok, total) in sorted(by_side.items()):
        print(f"  {side:6s} {ok}/{total}")

    # outcomes
    if not args.detection_only:
        print("\n=== field outcomes ===")
        print(f"{'pipeline':10s} {'correct':>8s} {'wrong':>7s} {'missing':>8s} "
              f"{'reject':>7s} {'accuracy':>9s} {'abstain':>8s} {'avg sec':>8s}")
        for pipeline in pipelines:
            counts = tally[pipeline]
            total = sum(counts[o] for o in OUTCOMES)
            if not total:
                continue
            accuracy = 100 * counts["correct"] / total
            # abstention: declined rather than guessed
            abstain = 100 * (counts["missing"] + counts["reject"]) / total
            print(f"{pipeline:10s} {counts['correct']:>8d} {counts['wrong']:>7d} "
                  f"{counts['missing']:>8d} {counts['reject']:>7d} "
                  f"{accuracy:>8.0f}% {abstain:>7.0f}% "
                  f"{counts['seconds']/len(rows):>8.0f}")

        for pipeline in pipelines:
            print(f"\n--- {pipeline}: per field ---")
            print(f"{'field':20s} " + " ".join(f"{o:>8s}" for o in OUTCOMES))
            for field_name in sorted(field_tally[pipeline]):
                counts = field_tally[pipeline][field_name]
                print(f"{field_name:20s} " +
                      " ".join(f"{counts[o]:>8d}" for o in OUTCOMES))

    # MRZ end to end
    if mrz_rows:
        with_mrz = [m for m in mrz_rows if m.get("has_mrz")]
        found = [m for m in with_mrz if m.get("lines_found")]
        parsed = [m for m in found if m.get("parsed")]
        print("\n=== MRZ, end to end ===")
        print(f"documents with an MRZ:     {len(with_mrz)}")
        print(f"zone located by OCR:       {len(found)}/{len(with_mrz)}")
        print(f"parsed successfully:       {len(parsed)}/{len(found) or 1}")
        if parsed:
            accuracy = sum(m["char_accuracy"] for m in parsed) / len(parsed)
            passed = sum(m["check_digits_passed"] for m in parsed)
            total = sum(m["check_digits_total"] for m in parsed)
            print(f"mean character accuracy:   {accuracy:.3f}")
            print(f"check digits passed:       {passed}/{total}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "testset": {"seed": manifest["seed"], "size": len(files)},
            "rows": rows,
            "totals": {p: dict(tally[p]) for p in pipelines},
            "per_field": {p: {k: dict(v) for k, v in field_tally[p].items()}
                          for p in pipelines},
            "mrz": mrz_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()