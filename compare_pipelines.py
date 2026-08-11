"""
Pipeline 1 (extract.py): detection -> OCR -> rules -> JSON
Pipeline 2 (donut_pipeline.py): image -> pretrained model -> JSON

    python compare_pipelines.py --limit 3
    python compare_pipelines.py --limit 5 --fields last_name date_of_birth
"""
import argparse
import glob
import json
import os
import time

import docxpand
from evaluate_docxpand import compare_fields


def run_classic(image_path):
    import extract
    started = time.time()
    result = extract.process_document(image_path)
    return result, time.time() - started


def run_donut(image_path, fields):
    import donut_pipeline
    started = time.time()
    result = donut_pipeline.process_document(image_path, fields=fields)
    return result, time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/docxpand/DocXPand-25k/documents")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--fields", nargs="*", default=None,
                        help="restrict Donut to these fields (it is slow)")
    parser.add_argument("--out", default="pipeline_comparison.json")
    args = parser.parse_args()

    labels_path = args.labels or docxpand.find_labels()
    if not labels_path:
        print("DocXPand labels not found — run download_docxpand.py --labels")
        return
    ground_truth = docxpand.GroundTruth(labels_path)

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = files[:args.limit]
    if not files:
        print(f"no images under {args.data}/")
        return

    print(f"comparing both pipelines on {len(files)} images\n")
    rows = []
    totals = {"classic": [0, 0, 0, 0.0], "donut": [0, 0, 0, 0.0]}

    for path in files:
        truth = ground_truth.fields(path)
        print(f"--- {os.path.basename(path)[:40]} ---")
        row = {"file": os.path.basename(path)}

        for label, runner in (("classic", lambda p: run_classic(p)),
                              ("donut", lambda p: run_donut(p, args.fields))):
            try:
                result, seconds = runner(path)
                correct, wrong, missing, per_field = compare_fields(
                    result["fields"], truth)
                totals[label][0] += correct
                totals[label][1] += wrong
                totals[label][2] += missing
                totals[label][3] += seconds
                row[label] = {
                    "correct": correct, "wrong": wrong, "missing": missing,
                    "seconds": round(seconds, 1),
                    "fields": {k: v for k, v in result["fields"].items() if v},
                    "per_field": per_field,
                }
                print(f"  {label:8s} correct={correct} wrong={wrong} "
                      f"missing={missing}  ({seconds:.0f}s)")
            except Exception as exc:
                row[label] = {"error": str(exc)}
                print(f"  {label:8s} ERROR: {exc}")

        rows.append(row)
        print()

    print("=== totals ===")
    print(f"{'pipeline':10s} {'correct':>8s} {'wrong':>7s} {'missing':>8s} {'avg sec':>9s}")
    for label, (correct, wrong, missing, seconds) in totals.items():
        total = correct + wrong + missing
        accuracy = f"{100*correct/total:.0f}%" if total else "-"
        print(f"{label:10s} {correct:>8d} {wrong:>7d} {missing:>8d} "
              f"{seconds/len(files):>9.0f}   ({accuracy})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "totals": totals}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()