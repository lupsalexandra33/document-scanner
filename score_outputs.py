"""
score_outputs.py — score saved pipeline outputs against ground truth.

Running OCR inside an evaluation loop exhausts memory on a machine with 8 GB,
so extraction is done one process per image (each exits and releases its
memory) and the JSON results are scored here afterwards. No OCR runs in this
script, so it finishes in seconds and can be re-run freely as the scoring rules
change.

Outcomes are split four ways rather than three:

    correct   the value matches ground truth
    wrong     a value was produced and it does not match
    missing   the document was localised, but no value was found for this field
    reject    the document was never localised, so extraction never ran

    python score_outputs.py --dir outputs_week5
"""
import argparse
import glob
import json
import os
import unicodedata
from collections import defaultdict

import docxpand
import mrz as mrz_module

OUTCOMES = ("correct", "wrong", "missing", "reject")


def normalise(text):
    """Strip accents, case and punctuation — those differences are formatting,
    not extraction errors ('MARECHAL DUBOIS' vs 'Maréchal-Dubois')."""
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c.lower() for c in text if c.isalnum()) or None


def classify(extracted, truth, detected):
    outcomes = {}
    for name, true_value in truth.items():
        if true_value is None:
            continue                        # not printed on this document
        got = extracted.get(name)
        if got is None:
            outcomes[name] = "missing" if detected else "reject"
        elif normalise(got) == normalise(true_value):
            outcomes[name] = "correct"
        else:
            outcomes[name] = "wrong"
    return outcomes


def score_mrz(result, ground_truth, image_path):
    # measure the MRZ stage end to end: located, then read, then parsed
    truth_lines = ground_truth.mrz(image_path)
    if not truth_lines:
        return None

    candidates = mrz_module.find_mrz_lines(result.get("ocr_texts", []))
    parsed = mrz_module.parse(candidates) if candidates else None

    entry = {
        "candidates": len(candidates),
        "parsed": parsed is not None,
    }
    if parsed:
        checks = parsed.get("checks", {})
        entry["checks_passed"] = sum(1 for v in checks.values() if v is True)
        entry["checks_total"] = len([v for v in checks.values() if v is not None])
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs_week5",
                        help="classic pipeline outputs")
    parser.add_argument("--donut-dir", default=None,
                        help="Donut outputs over the same test set; when given, "
                             "both pipelines are scored side by side")
    parser.add_argument("--testset", default="testset.json")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="final_evaluation.json")
    args = parser.parse_args()

    labels_path = args.labels or docxpand.find_labels()
    if not labels_path:
        print("DocXPand labels not found")
        return
    ground_truth = docxpand.GroundTruth(labels_path)

    with open(args.testset, encoding="utf-8") as f:
        image_paths = {os.path.basename(p).replace(".jpg", ""): p
                       for p in json.load(f)["files"]}

    rows, tally, field_tally, mrz_rows = [], defaultdict(int), defaultdict(lambda: defaultdict(int)), []

    for output_file in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        stem = os.path.basename(output_file).replace(".json", "")
        image_path = image_paths.get(stem)
        if image_path is None:
            continue

        result = json.load(open(output_file, encoding="utf-8"))
        detected = result.get("processing", {}).get("document_detected", False)
        outcomes = classify(result["fields"], ground_truth.fields(image_path), detected)

        for field, outcome in outcomes.items():
            tally[outcome] += 1
            field_tally[field][outcome] += 1

        rows.append({
            "file": stem,
            "class": ground_truth.document_class(image_path),
            "side": "front" if "front" in stem.lower() else "back",
            "detected": detected,
            "method": result.get("processing", {}).get("detection_method"),
            "outcomes": outcomes,
        })

        mrz_entry = score_mrz(result, ground_truth, image_path)
        if mrz_entry:
            mrz_entry["file"] = stem
            mrz_rows.append(mrz_entry)

    total = sum(tally[o] for o in OUTCOMES)
    if not total:
        print(f"no scorable outputs in {args.dir}/")
        return

    print(f"scored {len(rows)} documents, {total} fields\n")
    print("=== outcomes ===")
    for outcome in OUTCOMES:
        print(f"  {outcome:8s} {tally[outcome]:4d}  ({100*tally[outcome]/total:.0f}%)")
    print(f"\naccuracy   {100*tally['correct']/total:.0f}%")
    print(f"error rate {100*tally['wrong']/total:.0f}%")
    abstain = tally["missing"] + tally["reject"]
    print(f"abstention {100*abstain/total:.0f}%   "
          f"(declined rather than guessed)")

    answered = tally["correct"] + tally["wrong"]
    if answered:
        print(f"precision when it answers: {100*tally['correct']/answered:.0f}% "
              f"({tally['correct']}/{answered})")

    print("\n=== per field ===")
    print(f"{'field':20s} " + " ".join(f"{o:>8s}" for o in OUTCOMES))
    for field in sorted(field_tally):
        counts = field_tally[field]
        print(f"{field:20s} " + " ".join(f"{counts[o]:>8d}" for o in OUTCOMES))

    print("\n=== detection ===")
    by_side = defaultdict(lambda: [0, 0])
    for row in rows:
        by_side[row["side"]][1] += 1
        if row["detected"]:
            by_side[row["side"]][0] += 1
    for side, (ok, n) in sorted(by_side.items()):
        print(f"  {side:6s} {ok}/{n}")

    if mrz_rows:
        located = [m for m in mrz_rows if m["candidates"]]
        parsed = [m for m in mrz_rows if m["parsed"]]
        print("\n=== MRZ, end to end ===")
        print(f"documents with an MRZ:        {len(mrz_rows)}")
        print(f"candidate lines found by OCR: {len(located)}/{len(mrz_rows)}")
        print(f"parsed successfully:          {len(parsed)}/{len(mrz_rows)}")
        if parsed:
            passed = sum(m.get("checks_passed", 0) for m in parsed)
            checks = sum(m.get("checks_total", 0) for m in parsed)
            print(f"check digits passed:          {passed}/{checks}")

    # second pipeline, same test set
    comparison = None
    if args.donut_dir and os.path.isdir(args.donut_dir):
        donut_tally = defaultdict(int)
        donut_fields = defaultdict(lambda: defaultdict(int))
        donut_docs = 0

        for output_file in sorted(glob.glob(os.path.join(args.donut_dir, "*.json"))):
            stem = os.path.basename(output_file).replace(".json", "")
            image_path = image_paths.get(stem)
            if image_path is None:
                continue
            result = json.load(open(output_file, encoding="utf-8"))
            # Donut never uses detection, so nothing is ever rejected for it
            outcomes = classify(result["fields"], ground_truth.fields(image_path), True)
            for field, outcome in outcomes.items():
                donut_tally[outcome] += 1
                donut_fields[field][outcome] += 1
            donut_docs += 1

        donut_total = sum(donut_tally[o] for o in OUTCOMES)
        if donut_total:
            print(f"\n=== both pipelines, same {donut_docs} documents ===")
            print(f"{'pipeline':10s} {'correct':>8s} {'wrong':>7s} {'missing':>8s} "
                  f"{'reject':>7s} {'accuracy':>9s} {'abstain':>8s} {'precision':>10s}")

            # restrict the classic figures to the same documents, so the two
            # columns describe identical input
            classic_tally = defaultdict(int)
            donut_stems = {os.path.basename(f).replace(".json", "")
                           for f in glob.glob(os.path.join(args.donut_dir, "*.json"))}
            for row in rows:
                if row["file"] not in donut_stems:
                    continue
                for outcome in row["outcomes"].values():
                    classic_tally[outcome] += 1

            for label, counts in (("classic", classic_tally), ("donut", donut_tally)):
                subtotal = sum(counts[o] for o in OUTCOMES)
                if not subtotal:
                    continue
                answered = counts["correct"] + counts["wrong"]
                precision = f"{100*counts['correct']/answered:.0f}%" if answered else "-"
                abstain = counts["missing"] + counts["reject"]
                print(f"{label:10s} {counts['correct']:>8d} {counts['wrong']:>7d} "
                      f"{counts['missing']:>8d} {counts['reject']:>7d} "
                      f"{100*counts['correct']/subtotal:>8.0f}% "
                      f"{100*abstain/subtotal:>7.0f}% {precision:>10s}")

            comparison = {
                "documents": donut_docs,
                "classic": dict(classic_tally),
                "donut": dict(donut_tally),
                "donut_per_field": {k: dict(v) for k, v in donut_fields.items()},
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "documents": len(rows),
            "rows": rows,
            "totals": dict(tally),
            "per_field": {k: dict(v) for k, v in field_tally.items()},
            "mrz": mrz_rows,
            "comparison": comparison,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
